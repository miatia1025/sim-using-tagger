#!/bin/bash

# add, commit and push
git add .
echo -n "Commit message: "
read -r commit_msg
git commit -m "$commit_msg"
git push

# create commit url
remote_url=$(git remote get-url origin)
commit_hash=$(git log -1 --pretty=format:%H)
commit_url=$(echo $remote_url | sed -e "s/\.git.*$//")
commit_url="$commit_url/commit/$commit_hash"

# show commit url
echo $commit_url
