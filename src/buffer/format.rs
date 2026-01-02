//! Buffer format string parsing
//!
//! Parses Python buffer protocol format strings

use crate::buffer::BufferError;

/// Parsed format string
#[derive(Debug, Clone)]
pub struct FormatString {
    /// Endianness indicator (<, >, =, !)
    pub endian: Option<char>,
    /// Type character (b, B, h, H, i, I, l, L, q, Q, f, d)
    pub type_char: char,
    /// Item count (for arrays)
    pub count: Option<usize>,
}

impl FormatString {
    /// Parse a format string
    ///
    /// Supports format strings like "d", "<d", ">i", "=f", etc.
    pub fn parse(format: &str) -> Result<Self, BufferError> {
        let trimmed = format.trim();
        
        if trimmed.is_empty() {
            return Err(BufferError::InvalidFormat("Empty format string".to_string()));
        }
        
        let mut chars = trimmed.chars().peekable();
        
        // Parse endianness (optional)
        let endian = match chars.peek() {
            Some(&'<') | Some(&'>') | Some(&'=') | Some(&'!') => {
                chars.next()
            }
            _ => None,
        };
        
        // Parse type character
        let type_char = chars.next().ok_or_else(|| {
            BufferError::InvalidFormat("Missing type character".to_string())
        })?;
        
        // Parse count (optional)
        let count = if chars.peek().is_some() {
            let count_str: String = chars.collect();
            if !count_str.is_empty() {
                Some(count_str.parse().map_err(|_| {
                    BufferError::InvalidFormat(format!("Invalid count: {}", count_str))
                })?)
            } else {
                None
            }
        } else {
            None
        };
        
        // Validate type character
        match type_char {
            'b' | 'B' | 'h' | 'H' | 'i' | 'I' | 'l' | 'L' | 'q' | 'Q' | 'f' | 'd' => {}
            _ => {
                return Err(BufferError::InvalidFormat(
                    format!("Invalid type character: {}", type_char)
                ));
            }
        }
        
        Ok(FormatString {
            endian,
            type_char,
            count,
        })
    }
    
    /// Convert format string back to string
    #[allow(clippy::inherent_to_string)]
    pub fn to_string(&self) -> String {
        let mut result = String::new();
        if let Some(endian) = self.endian {
            result.push(endian);
        }
        result.push(self.type_char);
        if let Some(count) = self.count {
            result.push_str(&count.to_string());
        }
        result
    }
}

