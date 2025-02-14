# Run the tests
pytest
# Output 1 if they fail
if [[ $? -ge 1 ]]
then 
    exit 1
fi