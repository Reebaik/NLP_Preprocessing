import pandas as pd
import spacy

def perform_ner_and_extract_all(text):
    """
    Performs Named Entity Recognition (NER) on unstructured text and extracts
    all identified entities into a structured DataFrame.

    Args:
        text (str): The unstructured text data.

    Returns:
        pd.DataFrame: A DataFrame with 'Entity', 'Label', and 'Description' columns.
    """
    try:
        # Load the pre-trained NER model
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        print("Downloading 'en_core_web_sm' model...")
        spacy.cli.download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")

    # Process the text
    doc = nlp(text)

    # 1. Extract all identified entities
    extracted_data = []
    
    # Iterate through the entities in the document
    for ent in doc.ents:
        extracted_data.append({
            "Entity": ent.text,
            "Label": ent.label_,
            "Description": spacy.explain(ent.label_)
        })

    # 2. Create a DataFrame from the extracted data
    df = pd.DataFrame(extracted_data)
    
    return df

# --- Main script execution ---
if __name__ == "__main__":
    unstructured_text = """
    A press release from Apple Inc. in Cupertino, California, announced that CEO Tim Cook 
    will present the new iPhone 17 on September 25, 2026. The presentation will be held at 
    the Steve Jobs Theater. A recent study by Dr. Emily Carter and Dr. John Smith from 
    the National Aeronautics and Space Administration (NASA) revealed new data about Mars. 
    Their work was funded by a grant from the U.S. government, totaling $10 million. 
    The research, published in the journal Nature, is the culmination of three years of effort.
    """
    
    print("--- Unstructured Text Data ---")
    print(unstructured_text)
    
    # Perform the NER and extraction to get a comprehensive table
    structured_table = perform_ner_and_extract_all(unstructured_text)
    
    print("\n--- Extracted Structured Information (Full Table of Entities) ---")
    print(structured_table)