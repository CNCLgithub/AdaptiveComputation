for i in $(seq -f "%03g" 1 101)
do
	convert render/$i.png position_estimates/$i.png -append combined/combined_$i.png
done
convert combined/* -loop 0 -scale 500x500 combined.gif

