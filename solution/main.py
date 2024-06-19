import argparse
import logging
import random

import torch
import yaml
from torch import autocast
from torch.cuda.amp import GradScaler
from torch.utils.data import SequentialSampler
from torch.utils.tensorboard import SummaryWriter

from tqdm import trange
import torch as th
import einops

from models import MODELSecretSauceGenerator
from nn.dataset import AlbedoDataset, CustomDataset
from nn.loss import InnerProductLoss, compute_gradient_loss, prepare_kernels
from nn.network import CustomLoss
from nn.samplers import HardMiningSampler
from nn.util import *


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


def trn(config, model, device, train_loader, experiment_dir, tb_writer, num_epochs=1):
    if config['vgg_weight'] > 0:
        vgg_gram_loss = InnerProductLoss(config['vgg_layers'], device)

    if config['mode'] == 'bump':
        g_kernel, gd_kernel, gdd_kernel, sigma_space = prepare_kernels(device, config)

    # How often to plot results, every plot_results_N'th batch
    plot_results_N = config['plot_after']

    # First train a model for bump maps, and then for roughness maps
    for train_bump_maps_model in [True]:
        # Initialize a dictionary to store loss values
        losses = {'train_loss': []}

        # Model, Optimizer
        optimizer = optim.AdamW(model.parameters(), lr=float(config['lr']),
                                weight_decay=float(config['wd']))

        # Loss function is different depending on whether we're trainig a model for
        # bump maps or roughness maps

        # If it's roughness maps, we go with the default MSE
        rough_loss_module = nn.MSELoss()
        # If it's bump maps, we use our custom loss function
        bump_loss_module = CustomLoss()

        scaler = GradScaler()

        tqdm_out = TqdmToLogger(logger, level=logging.INFO)
        pbar = trange(num_epochs, file=tqdm_out, mininterval=3)
        for epoch in pbar:
            running_loss = 0.0
            for i, data in enumerate(train_loader):
                img_indices, data = data
                data = [d.to(device) for d in data]
                if config['rotate_batch']:
                    random_rot_k = random.randint(0, 3)
                    data = [torch.rot90(t, k=random_rot_k, dims=(-1, -2)) for t in data]
                inputs, bump, rough = data  # Ignoring rough maps here

                optimizer.zero_grad()

                with autocast(device_type='cuda', dtype=torch.float16):
                    outputs = model(inputs)

                    rough_loss = bump_custom = dloss = ddloss = dddloss = similarity_loss = 0
                    if config['mode'] == 'roughness':
                        rough_loss = rough_loss_module(outputs, rough)
                    elif config['mode'] == 'bump':
                        bump_custom = bump_loss_module(outputs, bump)

                        dloss = config['dloss_weight'] * compute_gradient_loss(bump, outputs, g_kernel, gd_kernel, sigma_space)
                        ddloss = config['ddloss_weight'] * compute_gradient_loss(bump, outputs, g_kernel, gdd_kernel, sigma_space)
                        dddloss = config['dddloss_weight'] * compute_gradient_loss(bump, outputs, gd_kernel, gdd_kernel, sigma_space)

                    if config['vgg_weight'] > 0:
                        targets = th.cat((rough, bump), dim=1)
                        vgg_targets = th.repeat_interleave(targets, 3, dim=1)
                        vgg_targets = th.cat((vgg_targets[:, :3], vgg_targets[:, 3:]), dim=0)
                        vgg_outputs = th.repeat_interleave(outputs, 3, dim=1)
                        vgg_outputs = th.cat((vgg_outputs[:, :3], vgg_outputs[:, 3:]), dim=0)

                        similarity_loss = config['vgg_weight'] * vgg_gram_loss(vgg_outputs, vgg_targets)

                    loss = dloss + ddloss + dddloss + rough_loss + similarity_loss

                    if config['sampler'] == 'hard_mining':
                        train_loader.sampler.update_with_local_losses(img_indices, loss)

                    if config['mode'] == 'roughness':
                        rough_loss = rough_loss.mean()
                    elif config['mode'] == 'bump':
                        dloss = dloss.mean()
                        ddloss = ddloss.mean()
                        dddloss = dddloss.mean()

                    loss = loss.mean()

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                # loss.backward()
                # optimizer.step()

                loss_dict = {'loss': loss.item()}
                if config['vgg_weight'] > 0:
                    loss_dict['sim'] = similarity_loss.item()

                tb_writer.add_scalar('trn/loss', loss.item(), (epoch * len(train_loader) + i) * config['batch_size'])
                if config['mode'] == 'roughness':
                    tb_writer.add_scalar('trn/rough', rough_loss.item(), (epoch * len(train_loader) + i) * config['batch_size'])
                    loss_dict['rou'] = rough_loss.item()
                elif config['mode'] == 'bump':
                    tb_writer.add_scalar('trn/bump_custom', bump_custom.item(), (epoch * len(train_loader) + i) * config['batch_size'])
                    tb_writer.add_scalar('trn/dloss', dloss.item(), (epoch * len(train_loader) + i) * config['batch_size'])
                    tb_writer.add_scalar('trn/ddloss', ddloss.item(), (epoch * len(train_loader) + i) * config['batch_size'])
                    tb_writer.add_scalar('trn/dddloss', dddloss.item(), (epoch * len(train_loader) + i) * config['batch_size'])
                    loss_dict['bum'] = bump_custom.item()
                    loss_dict['dlo'] = dloss.item()
                    loss_dict['ddl'] = ddloss.item()
                    loss_dict['ddd'] = dddloss.item()

                pbar.set_postfix(loss_dict)

                # Print statistics
                running_loss += float(loss.item())
                if i % plot_results_N == (plot_results_N - 1):  # Print every plot_results_N mini-batches
                    avg_loss = float(running_loss / float(plot_results_N))
                    if len(losses['train_loss']) and avg_loss < min(losses['train_loss']):
                        torch.save(model.state_dict(), os.path.join(experiment_dir, f'unet_{config["mode"]}_{epoch:05d}.pth'))
                    losses['train_loss'].append(avg_loss)
                    # compute minimum of list losses
                    if config['live_plot']:
                        live_plot(losses, title="{} Training Loss per {}'th Batch, ({} of {} epochs)".format(
                            "Bump Map" if train_bump_maps_model else "Rougness Map", plot_results_N, epoch + 1, num_epochs),
                                  x_label="Every {}'th Batch".format(plot_results_N), log_losses=False)
                    running_loss = 0.0

        # Save the model
        torch.save(model.state_dict(), os.path.join(experiment_dir, f"unet_{config['mode']}_last.pth"))

    logger.info('Finished Training')


def evaluate(model, root_dataset_evaluation_path, transform, config, device, predictions_dir,
             experiment_dir):
    # Get our evaluation dataset path
    evaluation_subset = config['subset_name']

    # Point towards the albedo images to be predicted
    pred_albedo_dirs = glob.glob(os.path.join(root_dataset_evaluation_path, evaluation_subset, "albedo-maps"))

    # Create the dataset & loader
    albedo_dataset = AlbedoDataset(pred_albedo_dirs, transform=transform)
    albedo_loader = DataLoader(albedo_dataset, batch_size=1, shuffle=False)  # Adjust batch size as needed

    # Now let's make some predictions, starting with the bump maps
    for test_bump_maps_model in [True]:

        # Load the appropriate model
        logger.info("Predicting maps:")
        model.load_state_dict(torch.load(os.path.join(experiment_dir, f"unet_{config['mode']}_last.pth"), map_location=device))
        model.eval()  # Set the model to evaluation mode

        with torch.no_grad():
            tqdm_out = TqdmToLogger(logger, level=logging.INFO)
            for idx, image in tqdm(enumerate(albedo_loader), total=len(albedo_loader), file=tqdm_out, mininterval=1):
                # Extract filename and common prefix
                path = pathlib.PurePath(albedo_dataset.albedo_image_paths[idx])

                common_prefix = albedo_basename(path.name)

                if common_prefix is None:
                    logger.info(albedo_dataset.albedo_image_paths[idx])
                    logger.info(path.name)

                # Generate prediction
                output = model(image.to(device))
                # Extract the vendor tag, and make sure the folder exists
                name, output_filename = "roughness-maps", roughness_filename(common_prefix)
                if config['mode'] == 'bump':
                    name, output_filename = "bump-maps", bump_filename(common_prefix)

                predictions_vendor_dir = os.path.join(predictions_dir, path.parent.parent.name, name)
                if not os.path.exists(predictions_vendor_dir):
                    os.makedirs(predictions_vendor_dir)

                # Call the save_prediction function
                save_prediction(output, output_filename, predictions_vendor_dir)  # Set is_bump_map to False for roughness maps


def create_submission(root_dir, predictions_dir, scripts_dir):
    # Now create the submission

    # Start by making our submission folder
    submission_dir = os.path.join(predictions_dir, "submission")

    # Delete any existing submission folder.
    if os.path.exists(submission_dir):
        shutil.rmtree(submission_dir)
    if os.path.exists(submission_dir):
        raise Exception("Submission path '{}' not deleted.".format(submission_dir))
    logger.info("Submission path '{}' successfully deleted.".format(submission_dir))

    # Now copy our prediction folder, this assumes
    shutil.copytree(predictions_dir, submission_dir)
    logger.info("Predictions '{}' successfully copied to submission folder '{}'.".format(predictions_dir, submission_dir))

    # Test that it's a directory and not a file
    if not os.path.isdir(submission_dir):
        raise Exception("Submission path '{}' is not a folder.".format(submission_dir))

    # Copy in our notebook to be submitted, make sure this is the one you want to be submitted and has been saved
    notebook_path = os.path.join(root_dir, "solution.ipynb")
    shutil.copy(notebook_path, submission_dir)

    solution_dir = os.path.join(submission_dir, 'solution')
    if os.path.isdir(solution_dir):
        shutil.rmtree(solution_dir)
    shutil.copytree(scripts_dir, solution_dir)

    # Test that it copied
    if not os.path.exists(os.path.join(submission_dir, os.path.basename(notebook_path))):
        raise Exception(
            "Error copying notebook '{}' to submission folder '{}' not empty.".format(notebook_path, submission_dir))
    logger.info("Notebook successfully '{}' to submission folder '{}'.".format(notebook_path, submission_dir))

    # Write the submission out to beside the submission folder
    submission_zip_folderpath = pathlib.PurePath(predictions_dir).parent

    # Get the current date and time
    current_time = datetime.now()

    # Format the filename
    submission_zip_filepath_formatted = os.path.join(submission_zip_folderpath,
                                                     current_time.strftime("submission-%d-%m-%Y-%H-%M-%S.zip"))

    # Create a ZipFile object in write mode
    logger.info("Compressing submission folder '{}' to zip file '{}'...".format(submission_dir,
                                                                          submission_zip_filepath_formatted))
    with zipfile.ZipFile(submission_zip_filepath_formatted, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Walk through each file in the folder
        for root, dirs, files in os.walk(submission_dir):
            for file in files:
                # Create a full path
                full_path = os.path.join(root, file)
                # Write the file to the zip, with a relative path
                zipf.write(full_path, os.path.relpath(full_path, os.path.join(submission_dir, '..')))
    logger.info("Done.")


def prepare_datasets(config, predictions_dir, root_dataset_training_path):
    trn_transform = transforms.Compose([
        transforms.ToTensor(),
        # Add other transformations as needed
    ])

    transform = transforms.Compose([
        transforms.ToTensor(),
        # Add more transforms as needed
    ])

    # Delete any existing predictions folder.
    if os.path.exists(predictions_dir):
        logger.warning("Predictions path '{}' not deleted.".format(predictions_dir))
    logger.info("Predictions path '{}' successfully deleted.".format(predictions_dir))
    # Find albedo, bump and roughness maps, allowing multiple vendors with the '*' wildcard

    subset_name = config['subset_name']
    albedo_dirs = glob.glob(os.path.join(root_dataset_training_path, subset_name, "albedo-maps"))
    bump_dirs = glob.glob(os.path.join(root_dataset_training_path, subset_name, "bump-maps"))
    rough_dirs = glob.glob(os.path.join(root_dataset_training_path, subset_name, "roughness-maps"))

    trn_dataset = CustomDataset(config, albedo_dirs, bump_dirs, rough_dirs, transform=trn_transform)

    if config['sampler'] == 'sequential':
        sampler = SequentialSampler(trn_dataset)
    elif config['sampler'] == 'random':
        sampler = None
    elif config['sampler'] == 'hard_mining':
        sampler = HardMiningSampler(trn_dataset)
    else:
        raise RuntimeError()

    train_loader = DataLoader(trn_dataset, batch_size=config['batch_size'],
                              sampler=sampler,
                              shuffle=config['shuffle'],
                              num_workers=config['num_threads'])

    return train_loader, transform


class CombinedModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.bump_model = MODELSecretSauceGenerator(config)
        self.roughness_model = MODELSecretSauceGenerator(config)

    def forward(self, x):
        bump = self.bump_model(x)
        roughness = self.roughness_model(x)
        return torch.cat((roughness, bump), dim=1)


def main():
    # add argparse here, first argument is path to yaml config
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("config", help="Path to yaml config file")
    argument_parser.add_argument("experiment_name", help="Experiment name")
    args = argument_parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    assert config['mode'] in ['roughness', 'bump']

    root_dir = os.path.join('/', os.sep, 'home', 'jupyter')
    root_dataset_path = os.path.join(root_dir, "dataset")
    experiments_dir = os.path.join('/', os.sep, 'home', 'team4', "experiments")
    scripts_dir = os.path.join('/', os.sep, 'home', 'team4', "chaos_hackaton_2023")

    root_dataset_evaluation_path = os.path.join(root_dataset_path, 'evaluation')

    experiment_dir = os.path.join(experiments_dir, args.experiment_name)
    if not os.path.isdir(experiment_dir):
        os.makedirs(experiment_dir)

    tb_writer = SummaryWriter(experiment_dir)

    root_dataset_training_path = os.path.join(root_dataset_path, 'training')
    predictions_dir = os.path.join(experiment_dir, 'predictions')  # Directory to save predictions

    # with open(os.path.join(experiment_dir, 'config.yml'), 'w') as f:
    #     yaml.dump(config, f)
    #
    # file_handler = logging.FileHandler(os.path.join(experiment_dir, 'log.txt'))
    # logger.addHandler(file_handler)
    #
    # # Check for GPU availability
    # if torch.backends.mps.is_available():
    #     device = torch.device("mps")
    # elif torch.cuda.is_available():
    #     device = torch.device("cuda")
    # else:
    #     device = torch.devide("cpu")
    # logger.info("Using device: %s", str(device))
    # logger.info("Process ID: %s", str(os.getpid()))
    # logger.info("Parent Process ID: %s", str(os.getppid()))
    #
    # train_loader, transform = prepare_datasets(config, predictions_dir, root_dataset_training_path)
    #
    # model = MODELSecretSauceGenerator(config)
    # model = model.to(device)
    #
    # trn(config, model, device, train_loader, experiment_dir, tb_writer, num_epochs=config['num_epochs'])
    #
    # evaluate(model, root_dataset_evaluation_path, transform, config, device, predictions_dir, experiment_dir)

    create_submission(root_dir, predictions_dir, scripts_dir)


if __name__ == '__main__':
    main()
