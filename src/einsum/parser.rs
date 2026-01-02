//! Einstein notation parser
//!
//! Parses einsum notation strings and extracts index information

use super::EinsumError;

/// Parsed einsum specification
#[derive(Debug, Clone)]
pub struct EinsumSpec {
    /// Input index labels for each input array
    pub input_labels: Vec<Vec<char>>,
    /// Output index labels
    pub output_labels: Vec<char>,
    /// All unique index labels
    pub all_labels: Vec<char>,
    /// Ellipsis positions (for broadcasting)
    pub ellipsis_positions: Vec<Option<usize>>,
}

/// Parse einsum notation string
///
/// Parses strings like "ij,jk->ik" or "i,i->" into an EinsumSpec
pub fn parse_einsum(subscripts: &str, num_inputs: usize) -> Result<EinsumSpec, EinsumError> {
    // Split into input and output parts
    let parts: Vec<&str> = subscripts.split("->").collect();
    
    if parts.is_empty() || parts.len() > 2 {
        return Err(EinsumError::ParseError(
            "Invalid einsum notation format".to_string()
        ));
    }
    
    // Parse input labels
    let input_part = parts[0].trim();
    let input_label_strs: Vec<&str> = input_part.split(',').map(|s| s.trim()).collect();
    
    if input_label_strs.len() != num_inputs {
        return Err(EinsumError::ParseError(
            format!("Expected {} input arrays, got {} labels", num_inputs, input_label_strs.len())
        ));
    }
    
    let mut input_labels = Vec::new();
    let mut ellipsis_positions = Vec::new();
    
    for label_str in input_label_strs {
        let mut labels = Vec::new();
        let mut ellipsis_pos = None;
        
        let chars: Vec<char> = label_str.chars().collect();
        let mut i = 0;
        while i < chars.len() {
            if i + 2 < chars.len() && chars[i] == '.' && chars[i + 1] == '.' && chars[i + 2] == '.' {
                // Found ellipsis
                if ellipsis_pos.is_some() {
                    return Err(EinsumError::ParseError("Multiple ellipsis in same input".to_string()));
                }
                ellipsis_pos = Some(labels.len());
                labels.push('.'); // Use '.' as marker for ellipsis position
                i += 3;
            } else if chars[i].is_alphabetic() || chars[i] == '_' {
                labels.push(chars[i]);
                i += 1;
            } else if chars[i].is_whitespace() {
                i += 1; // Skip whitespace
            } else {
                return Err(EinsumError::ParseError(
                    format!("Invalid character in einsum notation: {}", chars[i])
                ));
            }
        }
        
        input_labels.push(labels);
        ellipsis_positions.push(ellipsis_pos);
    }
    
    // Parse output labels (if specified)
    let output_labels = if parts.len() == 2 {
        let output_part = parts[1].trim();
        if output_part.is_empty() {
            // Implicit output - sum over all repeated indices
            Vec::new()
        } else {
            let mut labels = Vec::new();
            let chars: Vec<char> = output_part.chars().collect();
            let mut i = 0;
            while i < chars.len() {
                if i + 2 < chars.len() && chars[i] == '.' && chars[i + 1] == '.' && chars[i + 2] == '.' {
                    labels.push('.');
                    i += 3;
                } else if chars[i].is_alphabetic() || chars[i] == '_' {
                    labels.push(chars[i]);
                    i += 1;
                } else if chars[i].is_whitespace() {
                    i += 1;
                } else {
                    return Err(EinsumError::ParseError(
                        format!("Invalid character in output notation: {}", chars[i])
                    ));
                }
            }
            labels
        }
    } else {
        // No explicit output - will be determined implicitly
        Vec::new()
    };
    
    // Collect all unique labels
    let mut all_labels_set = std::collections::HashSet::new();
    for labels in &input_labels {
        for &label in labels {
            if label != '.' {
                all_labels_set.insert(label);
            }
        }
    }
    for &label in &output_labels {
        if label != '.' {
            all_labels_set.insert(label);
        }
    }
    
    let mut all_labels: Vec<char> = all_labels_set.into_iter().collect();
    all_labels.sort();
    
    Ok(EinsumSpec {
        input_labels,
        output_labels,
        all_labels,
        ellipsis_positions,
    })
}

