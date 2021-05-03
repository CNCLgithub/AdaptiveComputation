#!/usr/bin/env bash

WRKDIR="${1:-$PWD}"
echo $WRKDIR

rm -r ${WRKDIR}concat_images/
mkdir  ${WRKDIR}concat_images/
rm -r ${WRKDIR}resized_obs_rf_masks/
mkdir ${WRKDIR}resized_obs_rf_masks/
rm -r ${WRKDIR}resized_pred_dist_rf_masks/
mkdir ${WRKDIR}resized_pred_dist_rf_masks/

for file in $(find ${WRKDIR}render -type f -exec basename {} \; | sort)
do
    echo $file
    convert ${WRKDIR}obs_rf_masks/$file -resize x800 ${WRKDIR}resized_obs_rf_masks/$file
    convert ${WRKDIR}pred_dist_rf_masks/$file -resize x800 ${WRKDIR}resized_pred_dist_rf_masks/$file
    convert +append ${WRKDIR}render/$file ${WRKDIR}resized_obs_rf_masks/$file ${WRKDIR}resized_pred_dist_rf_masks/$file ${WRKDIR}concat_images/$file
done

#convert ${WRKDIR}concat_images/* ${WRKDIR}concat_receptive_fields.gif
./scripts/figures/png_to_gif.sh ${WRKDIR}concat_images
