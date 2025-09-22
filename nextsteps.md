# next steps
## code to test responses between models
example workflow:
(optional) write a system prompt
write a prompt
have the prompt with optional system prompt run through a list of different ollama models of the same quant size, i.e. approx 4b
save the resulting response to an excel file along with the model name, system prompt if used, and prompt.
repeat mutliple times if needed and then repeat with multiple models with each row being a new itteration or a new model
then run through the existing code using gemma2, because it has already been shown to be accurate and reliable, to grade the responses on how well each response responds to the prompt.
compare grades for each model

## code to evaluate images and provide coded results
run images through the code with a specific critera such as for the first digit, if it is a one story building say 1, two story say 2 and so on, then for the second digit, if it is neoclasical say 1, brutalist say 2, and so on, and repeat to go through all relevant criteria and get a single number with different values at each digit with information about the buildings which can then be fed into grasshopper to generate the neede buildings given the footprint. 

Isabel Patworowski does not think architecture can be analyzed in a binary way with ai so that I can generate buildings with grasshopper from a list of features. 

Is it better to have low fidelity street scapes from google maps or open street maps lidar data that are true to life despite their low fidelity, or high fidelity generic buildings based on image analysis and grasshopper generated buildings?

## extract words and code with a lookup table
run the existing code asking for a single word, or potentially multiple words in the case of multiple themes, in alphabetical order, and use a lookup table to translate the words into values to run the same anova, and t-tests