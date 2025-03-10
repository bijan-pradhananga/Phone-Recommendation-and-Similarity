from fuzzywuzzy import process

def find_closest_match(phone_model, index):
    match = process.extractOne(phone_model, index)
    if match is None:
        return None
    best_match, score = match
    return best_match if score > 75 else None