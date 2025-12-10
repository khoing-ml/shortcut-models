#1 /usr/bin/bash

## get arguments 
message=$1
branch_name=$2


git add .
git commit -m "$message"
git push origin $branch_name