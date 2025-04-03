#!/usr/bin/env python3
"""
Script to count specific conversation pairs in a JSON file.
Specifically looks for "What objects are visible in the video?" questions.
"""

import json
import os
import re
from typing import List, Dict, Any

def count_conversation_pairs(file_path: str) -> int:
    """
    Count the number of conversation pairs where:
    1. Human asks "What objects are visible in the video?"
    2. GPT responds with details about visible objects
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        int: Number of matching conversation pairs
    """
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' does not exist.")
        return -1
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # Counter for the specific conversation pairs
        pair_count = 0
        sample_count = 0

        # Process each item in the dataset
        if isinstance(data, list):
            # If the data is a list of conversations
            for item in data:
                sample_count += 1
                conversations = item.get("conversations", [])
                pair_count += count_object_pairs_in_conversation(conversations)
                
        elif isinstance(data, dict):
            # If the data is a dictionary with a list of items
            for key, item in data.items():
                if isinstance(item, dict) and "conversations" in item:
                    sample_count += 1
                    conversations = item.get("conversations", [])
                    pair_count += count_object_pairs_in_conversation(conversations)
            
            # Alternative format: if the data is a single item with conversations
            if "conversations" in data:
                sample_count += 1
                conversations = data.get("conversations", [])
                pair_count += count_object_pairs_in_conversation(conversations)
                
        return pair_count, sample_count
    
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        return -1, -1

def count_object_pairs_in_conversation(conversations: List[Dict[str, Any]]) -> int:
    """
    Count the object-related Q&A pairs in a single conversation.
    
    Args:
        conversations: List of conversation turns
        
    Returns:
        int: Number of object-related Q&A pairs
    """
    count = 0
    
    # Iterate through conversation turns looking for the pattern
    for i in range(len(conversations) - 1):
        current = conversations[i]
        next_turn = conversations[i + 1]
        
        # Check if current is human asking about objects
        if (current.get("from") == "human" and 
            "What objects are visible in the video?" in current.get("value", "")):
            
            # Check if next turn is gpt responding about objects
            if next_turn.get("from") == "gpt":
                count += 1
    
    return count

def main():
    file_path = "/map-vepfs/huggingface/datasets/llava_video_178k/0_30_s_academic_v0_1/0_30_s_academic_oe_v0_1_qa_processed.json"
    
    pairs_count, total_samples = count_conversation_pairs(file_path)
    
    if pairs_count >= 0:
        print(f"The JSON file contains {pairs_count} conversation pairs about 'What objects are visible in the video?'")
        print(f"Total conversation samples in the file: {total_samples}")
    
if __name__ == "__main__":
    main()