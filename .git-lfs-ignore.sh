#! /bin/bash
echo 'Adding files larger than 99M to .gitignore'
find . -size +99M | cat >> .gitignore
echo 'Done'
