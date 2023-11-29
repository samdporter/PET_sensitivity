# Deletes all temporary files recursively in the directory 
# run with sh delete_tmp_files.sh or ./delete_tmp_files.sh

find . -name "*tmp_*.hs" -exec rm -rf {} \;
find . -name "*tmp_*.s" -exec rm -rf {} \;