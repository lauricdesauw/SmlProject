# !/bin/bash 
echo "review = ["
for f in *; do 
echo "\"" 
cat $f 
echo "\","
done 
echo "]"
