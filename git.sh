#1 /usr/bin/bash

## get arguments 
message=$1

git add .
git commit -m "$message"
git push origin LFM