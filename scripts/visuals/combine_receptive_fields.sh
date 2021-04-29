#!/usr/bin/env bash

WRKDIR="${1:-$PWD}"
echo $WRKDIR

rm -r ${WRKDIR}concat_images/
mkdir  ${WRKDIR}concat_images/
rm -r ${WRKDIR}rf_mask_distributions/
mkdir ${WRKDIR}rf_mask_distributions/

for file in $(find ${WRKDIR}render -type f -exec basename {} \; | sort)
do
    echo $file
    convert ${WRKDIR}mask_distributions/$file -resize x800 ${WRKDIR}rf_mask_distributions/$file
    convert +append ${WRKDIR}render/$file ${WRKDIR}rf_mask_distributions/$file ${WRKDIR}concat_images/$file
done

#convert ${WRKDIR}concat_images/* ${WRKDIR}concat_receptive_fields.gif
#./scripts/figures/png_to_gif.sh ${WRKDIR}concat_images
