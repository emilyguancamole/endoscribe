from word2number import w2n
import roman
import contractions

def numbers_to_digits(input_string):
    number_words = {
        "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
        "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
        "eighteen": 18, "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70,
        "eighty": 80, "ninety": 90, "hundred": 100
    }

    words = input_string.split()
    result = [] 
    current_number = 0 
    for word in words:
        if word in number_words:
            if word=="hundred" or word=="thousand":
                if current_number == 0:
                    current_number = 1  # Handle cases like one hundred by multiplying by scale
                current_number *= number_words[word] 
            else:
                current_number += number_words[word]  # Add value of number word
        # For some reason parakeet randomly outputs some numbers in roman numerals, so check for those
        elif word!='i' and all(char in "ivxlcdm" for char in word):  # If all characters in word are Roman chars #! would there ever be an individual x that isn't a number?
            try:
                roman_value = roman.fromRoman(word.upper())
                current_number += roman_value
            except roman.InvalidRomanNumeralError:
                result.append(word)  # Append unrecognized Roman numeral
        else:
            # Reach a word that isn't a number, so finalize current number
            if current_number > 0:
                result.append(str(current_number))
                current_number = 0  
            result.append(word)
    if current_number > 0:
        result.append(str(current_number))

    return ' '.join(result)

def process_predictions(pred_file, transcript_col_name="pred_transcript"):
    ''' Process predictions to be lowercase, remove punctuation, expand contractions, make numbers digits.
        For evaluation of WER only.
    '''
    pred_df = pd.read_csv(pred_file)
    pred_df[transcript_col_name] = pred_df[transcript_col_name].apply(lambda x: x.lower()) # lowercase
    pred_df[transcript_col_name] = pred_df[transcript_col_name].apply(lambda x: contractions.fix(x)) # contractions
    # pred_df[transcript_col_name] = pred_df[transcript_col_name].apply(lambda x: x.strip("[]()'")) # especially parakeet seems to keep these characters as part of the transciption
    # Convert numbers to digits
    pred_df[transcript_col_name] = pred_df[transcript_col_name].apply(lambda x: numbers_to_digits(x))

    pred_df[transcript_col_name] = pred_df[transcript_col_name].apply(lambda x: re.sub(r"[^\w\s]", "", str(x))) # punctuation
    # pred_df[transcript_col_name] = pred_df[transcript_col_name].apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)))

    # for x in pred_df[transcript_col_name]: print(x)
    
    return pred_df