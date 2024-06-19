#! /bin/bash

IS_SUBMISSION="submission/"

SOURCE_DIR="/home/team4/experiments/all_2023-11-25_13-04-20/predictions/${IS_SUBMISSION}"
TARGET_DIR="/home/team4/experiments/all_bump_2023-11-26_17-08-38/predictions/${IS_SUBMISSION}"

for VENDOR_DIR in "${SOURCE_DIR}/"*; do
  # if the vendor dir is not a directory, skip it
  if [ ! -d "${VENDOR_DIR}" ]; then
    continue
  fi

  VENDOR_DIRNAME=$(basename "${VENDOR_DIR}")

  if [[ $VENDOR_DIRNAME == "submission" ]] || [[ $VENDOR_DIRNAME == "solution" ]]; then
    continue
  fi

  echo "copying vendor ${VENDOR_DIR}, ${VENDOR_DIRNAME}"


  rsync -avz "${VENDOR_DIR}/roughness-maps" "${TARGET_DIR}/${VENDOR_DIRNAME}"


done
