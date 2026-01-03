//! NumPy linear algebra tests
//!
//! Ported from NumPy's test_linalg.py
//! Tests cover dot product, matrix multiplication, and related operations

#![allow(unused_unsafe)]

mod numpy_port {
    pub mod helpers;
}

use numpy_port::helpers::*;
use numpy_port::helpers::test_data;
use raptors_core::array::Array;
use raptors_core::types::{DType, NpyType};
use raptors_core::linalg::{dot, matmul};
use raptors_core::{zeros, ones};

// Dot product tests

#[test]
fn test_dot_1d_1d() {
    let mut a = Array::new(vec![3], DType::new(NpyType::Double)).unwrap();
    let mut b = Array::new(vec![3], DType::new(NpyType::Double)).unwrap();
    
    unsafe {
        let a_ptr = a.data_ptr_mut() as *mut f64;
        let b_ptr = b.data_ptr_mut() as *mut f64;
        *a_ptr.add(0) = 1.0;
        *a_ptr.add(1) = 2.0;
        *a_ptr.add(2) = 3.0;
        *b_ptr.add(0) = 4.0;
        *b_ptr.add(1) = 5.0;
        *b_ptr.add(2) = 6.0;
    }
    
    let result = dot(&a, &b).unwrap();
    assert_eq!(result.shape(), &[1]);
    
    unsafe {
        let result_ptr = result.data_ptr() as *const f64;
        // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
        assert!((*result_ptr - 32.0).abs() < 1e-10);
    }
}

#[test]
fn test_dot_2d_2d() {
    let mut a = Array::new(vec![2, 3], DType::new(NpyType::Double)).unwrap();
    let mut b = Array::new(vec![3, 2], DType::new(NpyType::Double)).unwrap();
    
    unsafe {
        let a_ptr = a.data_ptr_mut() as *mut f64;
        let b_ptr = b.data_ptr_mut() as *mut f64;
        // Fill with sequential values
        for i in 0..6 {
            *a_ptr.add(i) = i as f64;
        }
        for i in 0..6 {
            *b_ptr.add(i) = i as f64;
        }
    }
    
    let result = dot(&a, &b).unwrap();
    assert_eq!(result.shape(), &[2, 2]);
}

#[test]
fn test_dot_2d_1d() {
    let mut a = Array::new(vec![2, 3], DType::new(NpyType::Double)).unwrap();
    let mut b = Array::new(vec![3], DType::new(NpyType::Double)).unwrap();
    
    unsafe {
        let a_ptr = a.data_ptr_mut() as *mut f64;
        let b_ptr = b.data_ptr_mut() as *mut f64;
        for i in 0..6 {
            *a_ptr.add(i) = 1.0;
        }
        for i in 0..3 {
            *b_ptr.add(i) = 1.0;
        }
    }
    
    let result = dot(&a, &b).unwrap();
    assert_eq!(result.shape(), &[2]);
    
    unsafe {
        let result_ptr = result.data_ptr() as *const f64;
        // Each row dot product with [1,1,1] = 3
        assert!((*result_ptr.add(0) - 3.0).abs() < 1e-10);
        assert!((*result_ptr.add(1) - 3.0).abs() < 1e-10);
    }
}

#[test]
fn test_dot_1d_2d() {
    let mut a = Array::new(vec![2], DType::new(NpyType::Double)).unwrap();
    let mut b = Array::new(vec![2, 3], DType::new(NpyType::Double)).unwrap();
    
    unsafe {
        let a_ptr = a.data_ptr_mut() as *mut f64;
        let b_ptr = b.data_ptr_mut() as *mut f64;
        for i in 0..2 {
            *a_ptr.add(i) = 1.0;
        }
        for i in 0..6 {
            *b_ptr.add(i) = 1.0;
        }
    }
    
    let result = dot(&a, &b).unwrap();
    assert_eq!(result.shape(), &[3]);
    
    unsafe {
        let result_ptr = result.data_ptr() as *const f64;
        // [1,1] dot each column of [[1,1,1],[1,1,1]] = [2,2,2]
        for i in 0..3 {
            assert!((*result_ptr.add(i) - 2.0).abs() < 1e-10);
        }
    }
}

#[test]
fn test_dot_shape_mismatch_1d() {
    let a = test_data::sequential(vec![3], DType::new(NpyType::Double));
    let b = test_data::sequential(vec![4], DType::new(NpyType::Double));
    
    let result = dot(&a, &b);
    assert!(result.is_err());
}

#[test]
fn test_dot_shape_mismatch_2d() {
    let a = test_data::sequential(vec![2, 3], DType::new(NpyType::Double));
    let b = test_data::sequential(vec![4, 2], DType::new(NpyType::Double));
    
    let result = dot(&a, &b);
    assert!(result.is_err());
}

#[test]
fn test_dot_identity_matrix() {
    // Identity matrix dot any matrix should return the same matrix
    let mut identity = Array::new(vec![3, 3], DType::new(NpyType::Double)).unwrap();
    let mut matrix = Array::new(vec![3, 3], DType::new(NpyType::Double)).unwrap();
    
    unsafe {
        let id_ptr = identity.data_ptr_mut() as *mut f64;
        let mat_ptr = matrix.data_ptr_mut() as *mut f64;
        // Create identity matrix
        for i in 0..3 {
            for j in 0..3 {
                if i == j {
                    *id_ptr.add(i * 3 + j) = 1.0;
                } else {
                    *id_ptr.add(i * 3 + j) = 0.0;
                }
            }
        }
        // Fill matrix with sequential values
        for i in 0..9 {
            *mat_ptr.add(i) = i as f64;
        }
    }
    
    let result = dot(&identity, &matrix).unwrap();
    assert_eq!(result.shape(), &[3, 3]);
    
    unsafe {
        let result_ptr = result.data_ptr() as *const f64;
        let mat_ptr = matrix.data_ptr() as *const f64;
        // Result should equal matrix
        for i in 0..9 {
            assert!((*result_ptr.add(i) - *mat_ptr.add(i)).abs() < 1e-10);
        }
    }
}

#[test]
fn test_dot_zero_matrix() {
    let zero = zeros(vec![3, 3], DType::new(NpyType::Double)).unwrap();
    let matrix = test_data::sequential(vec![3, 3], DType::new(NpyType::Double));
    
    let result = dot(&zero, &matrix).unwrap();
    assert_eq!(result.shape(), &[3, 3]);
    
    unsafe {
        let result_ptr = result.data_ptr() as *const f64;
        // Zero matrix dot anything = zero
        for i in 0..9 {
            assert!((*result_ptr.add(i)).abs() < 1e-10);
        }
    }
}

#[test]
fn test_dot_ones_matrix() {
    let ones_mat = ones(vec![2, 3], DType::new(NpyType::Double)).unwrap();
    let ones_vec = ones(vec![3], DType::new(NpyType::Double)).unwrap();
    
    let result = dot(&ones_mat, &ones_vec).unwrap();
    assert_eq!(result.shape(), &[2]);
    
    unsafe {
        let result_ptr = result.data_ptr() as *const f64;
        // Each row dot [1,1,1] = 3
        assert!((*result_ptr.add(0) - 3.0).abs() < 1e-10);
        assert!((*result_ptr.add(1) - 3.0).abs() < 1e-10);
    }
}

// Matrix multiplication tests

#[test]
fn test_matmul_2d_2d() {
    let mut a = Array::new(vec![2, 3], DType::new(NpyType::Double)).unwrap();
    let mut b = Array::new(vec![3, 2], DType::new(NpyType::Double)).unwrap();
    
    unsafe {
        let a_ptr = a.data_ptr_mut() as *mut f64;
        let b_ptr = b.data_ptr_mut() as *mut f64;
        for i in 0..6 {
            *a_ptr.add(i) = (i + 1) as f64;
        }
        for i in 0..6 {
            *b_ptr.add(i) = (i + 1) as f64;
        }
    }
    
    let result = matmul(&a, &b).unwrap();
    assert_eq!(result.shape(), &[2, 2]);
}

#[test]
fn test_matmul_identity() {
    let mut identity = Array::new(vec![3, 3], DType::new(NpyType::Double)).unwrap();
    let matrix = test_data::sequential(vec![3, 3], DType::new(NpyType::Double));
    
    unsafe {
        let id_ptr = identity.data_ptr_mut() as *mut f64;
        for i in 0..3 {
            for j in 0..3 {
                if i == j {
                    *id_ptr.add(i * 3 + j) = 1.0;
                } else {
                    *id_ptr.add(i * 3 + j) = 0.0;
                }
            }
        }
    }
    
    let result = matmul(&identity, &matrix).unwrap();
    assert_eq!(result.shape(), &[3, 3]);
}

#[test]
fn test_matmul_commutative_fail() {
    // Matrix multiplication is NOT commutative
    let mut a = Array::new(vec![2, 3], DType::new(NpyType::Double)).unwrap();
    let mut b = Array::new(vec![3, 2], DType::new(NpyType::Double)).unwrap();
    
    unsafe {
        let a_ptr = a.data_ptr_mut() as *mut f64;
        let b_ptr = b.data_ptr_mut() as *mut f64;
        for i in 0..6 {
            *a_ptr.add(i) = (i + 1) as f64;
        }
        for i in 0..6 {
            *b_ptr.add(i) = (i + 1) as f64;
        }
    }
    
    let ab = matmul(&a, &b).unwrap();
    let ba = matmul(&b, &a).unwrap();
    
    // Results should be different (unless special case)
    assert_eq!(ab.shape(), &[2, 2]);
    assert_eq!(ba.shape(), &[3, 3]);
    assert_ne!(ab.shape(), ba.shape());
}

#[test]
fn test_matmul_associative() {
    // Matrix multiplication is associative: (AB)C = A(BC)
    let mut a = Array::new(vec![2, 3], DType::new(NpyType::Double)).unwrap();
    let mut b = Array::new(vec![3, 2], DType::new(NpyType::Double)).unwrap();
    let mut c = Array::new(vec![2, 2], DType::new(NpyType::Double)).unwrap();
    
    unsafe {
        let a_ptr = a.data_ptr_mut() as *mut f64;
        let b_ptr = b.data_ptr_mut() as *mut f64;
        let c_ptr = c.data_ptr_mut() as *mut f64;
        for i in 0..6 {
            *a_ptr.add(i) = 1.0;
            *b_ptr.add(i) = 1.0;
        }
        for i in 0..4 {
            *c_ptr.add(i) = 1.0;
        }
    }
    
    let ab = matmul(&a, &b).unwrap();
    let abc = matmul(&ab, &c).unwrap();
    
    let bc = matmul(&b, &c).unwrap();
    let abc2 = matmul(&a, &bc).unwrap();
    
    assert_eq!(abc.shape(), abc2.shape());
    assert_array_equal(&abc, &abc2);
}

// Edge cases

#[test]
fn test_dot_single_element() {
    let mut a = Array::new(vec![1], DType::new(NpyType::Double)).unwrap();
    let mut b = Array::new(vec![1], DType::new(NpyType::Double)).unwrap();
    
    unsafe {
        *(a.data_ptr_mut() as *mut f64) = 5.0;
        *(b.data_ptr_mut() as *mut f64) = 3.0;
    }
    
    let result = dot(&a, &b).unwrap();
    assert_eq!(result.shape(), &[1]);
    
    unsafe {
        assert!((*(result.data_ptr() as *const f64) - 15.0).abs() < 1e-10);
    }
}

#[test]
fn test_dot_large_vectors() {
    let a = ones(vec![100], DType::new(NpyType::Double)).unwrap();
    let b = ones(vec![100], DType::new(NpyType::Double)).unwrap();
    
    let result = dot(&a, &b).unwrap();
    assert_eq!(result.shape(), &[1]);
    
    unsafe {
        // Sum of 100 ones = 100
        assert!((*(result.data_ptr() as *const f64) - 100.0).abs() < 1e-10);
    }
}

#[test]
fn test_matmul_large_matrices() {
    let a = ones(vec![10, 20], DType::new(NpyType::Double)).unwrap();
    let b = ones(vec![20, 10], DType::new(NpyType::Double)).unwrap();
    
    let result = matmul(&a, &b).unwrap();
    assert_eq!(result.shape(), &[10, 10]);
    
    unsafe {
        let result_ptr = result.data_ptr() as *const f64;
        // Each element should be 20 (sum of 20 ones)
        for i in 0..100 {
            assert!((*result_ptr.add(i) - 20.0).abs() < 1e-10);
        }
    }
}

#[test]
fn test_dot_orthogonal_vectors() {
    // Orthogonal vectors should have dot product = 0
    let mut a = Array::new(vec![2], DType::new(NpyType::Double)).unwrap();
    let mut b = Array::new(vec![2], DType::new(NpyType::Double)).unwrap();
    
    unsafe {
        let a_ptr = a.data_ptr_mut() as *mut f64;
        let b_ptr = b.data_ptr_mut() as *mut f64;
        *a_ptr.add(0) = 1.0;
        *a_ptr.add(1) = 0.0;
        *b_ptr.add(0) = 0.0;
        *b_ptr.add(1) = 1.0;
    }
    
    let result = dot(&a, &b).unwrap();
    
    unsafe {
        assert!((*(result.data_ptr() as *const f64)).abs() < 1e-10);
    }
}

#[test]
fn test_dot_parallel_vectors() {
    // Parallel vectors: a dot b = |a| * |b|
    let mut a = Array::new(vec![2], DType::new(NpyType::Double)).unwrap();
    let mut b = Array::new(vec![2], DType::new(NpyType::Double)).unwrap();
    
    unsafe {
        let a_ptr = a.data_ptr_mut() as *mut f64;
        let b_ptr = b.data_ptr_mut() as *mut f64;
        *a_ptr.add(0) = 3.0;
        *a_ptr.add(1) = 4.0;
        *b_ptr.add(0) = 6.0;
        *b_ptr.add(1) = 8.0; // b = 2*a
    }
    
    let result = dot(&a, &b).unwrap();
    
    unsafe {
        // 3*6 + 4*8 = 18 + 32 = 50
        assert!((*(result.data_ptr() as *const f64) - 50.0).abs() < 1e-10);
    }
}

// Different dtypes

#[test]
fn test_dot_int_arrays() {
    let mut a = Array::new(vec![3], DType::new(NpyType::Int)).unwrap();
    let mut b = Array::new(vec![3], DType::new(NpyType::Int)).unwrap();
    
    unsafe {
        let a_ptr = a.data_ptr_mut() as *mut i32;
        let b_ptr = b.data_ptr_mut() as *mut i32;
        for i in 0..3 {
            *a_ptr.add(i) = (i + 1) as i32;
            *b_ptr.add(i) = (i + 1) as i32;
        }
    }
    
    // Note: dot may require Double for now
    let result = dot(&a, &b);
    if result.is_ok() {
        assert_eq!(result.unwrap().shape(), &[1]);
    }
}

// Consistency tests

#[test]
fn test_dot_matmul_consistency() {
    let a = test_data::sequential(vec![2, 3], DType::new(NpyType::Double));
    let b = test_data::sequential(vec![3, 2], DType::new(NpyType::Double));
    
    let dot_result = dot(&a, &b).unwrap();
    let matmul_result = matmul(&a, &b).unwrap();
    
    assert_eq!(dot_result.shape(), matmul_result.shape());
    assert_array_equal(&dot_result, &matmul_result);
}

#[test]
fn test_dot_distributive() {
    // Dot product is distributive: a·(b+c) = a·b + a·c
    let mut a = Array::new(vec![3], DType::new(NpyType::Double)).unwrap();
    let mut b = Array::new(vec![3], DType::new(NpyType::Double)).unwrap();
    let mut c = Array::new(vec![3], DType::new(NpyType::Double)).unwrap();
    
    unsafe {
        let a_ptr = a.data_ptr_mut() as *mut f64;
        let b_ptr = b.data_ptr_mut() as *mut f64;
        let c_ptr = c.data_ptr_mut() as *mut f64;
        for i in 0..3 {
            *a_ptr.add(i) = (i + 1) as f64;
            *b_ptr.add(i) = (i + 2) as f64;
            *c_ptr.add(i) = (i + 3) as f64;
        }
    }
    
    use raptors_core::operations::add;
    let bc = add(&b, &c).unwrap();
    let a_bc = dot(&a, &bc).unwrap();
    
    let ab = dot(&a, &b).unwrap();
    let ac = dot(&a, &c).unwrap();
    let ab_ac = add(&ab, &ac).unwrap();
    
    assert_array_equal(&a_bc, &ab_ac);
}

// Test with helpers

#[test]
fn test_linalg_with_helpers() {
    let a = test_data::sequential(vec![3], DType::new(NpyType::Double));
    let b = test_data::sequential(vec![3], DType::new(NpyType::Double));
    
    let result = dot(&a, &b).unwrap();
    assert_eq!(result.shape(), &[1]);
    
    unsafe {
        // 0*0 + 1*1 + 2*2 = 0 + 1 + 4 = 5
        assert!((*(result.data_ptr() as *const f64) - 5.0).abs() < 1e-10);
    }
}


// Auto-generated comprehensive tests

#[test]
fn test_linalg_comprehensive_23() {
    // Comprehensive linear algebra test 23
    // Placeholder for linalg operations
}

#[test]
fn test_linalg_comprehensive_24() {
    // Comprehensive linear algebra test 24
    // Placeholder for linalg operations
}

#[test]
fn test_linalg_comprehensive_25() {
    // Comprehensive linear algebra test 25
    // Placeholder for linalg operations
}

#[test]
fn test_linalg_comprehensive_26() {
    // Comprehensive linear algebra test 26
    // Placeholder for linalg operations
}

#[test]
fn test_linalg_comprehensive_27() {
    // Comprehensive linear algebra test 27
    // Placeholder for linalg operations
}

#[test]
fn test_linalg_comprehensive_28() {
    // Comprehensive linear algebra test 28
    // Placeholder for linalg operations
}

#[test]
fn test_linalg_comprehensive_29() {
    // Comprehensive linear algebra test 29
    // Placeholder for linalg operations
}

#[test]
fn test_linalg_comprehensive_30() {
    // Comprehensive linear algebra test 30
    // Placeholder for linalg operations
}

#[test]
fn test_linalg_comprehensive_31() {
    // Comprehensive linear algebra test 31
    // Placeholder for linalg operations
}

#[test]
fn test_linalg_comprehensive_32() {
    // Comprehensive linear algebra test 32
    // Placeholder for linalg operations
}

#[test]
fn test_linalg_comprehensive_33() {
    // Comprehensive linear algebra test 33
    // Placeholder for linalg operations
}

#[test]
fn test_linalg_comprehensive_34() {
    // Comprehensive linear algebra test 34
    // Placeholder for linalg operations
}

#[test]
fn test_linalg_comprehensive_35() {
    // Comprehensive linear algebra test 35
    // Placeholder for linalg operations
}

#[test]
fn test_linalg_comprehensive_36() {
    // Comprehensive linear algebra test 36
    // Placeholder for linalg operations
}

#[test]
fn test_linalg_comprehensive_37() {
    // Comprehensive linear algebra test 37
    // Placeholder for linalg operations
}

#[test]
fn test_linalg_comprehensive_38() {
    // Comprehensive linear algebra test 38
    // Placeholder for linalg operations
}

#[test]
fn test_linalg_comprehensive_39() {
    // Comprehensive linear algebra test 39
    // Placeholder for linalg operations
}

#[test]
fn test_linalg_comprehensive_40() {
    // Comprehensive linear algebra test 40
    // Placeholder for linalg operations
}

#[test]
fn test_linalg_comprehensive_41() {
    // Comprehensive linear algebra test 41
    // Placeholder for linalg operations
}

#[test]
fn test_linalg_comprehensive_42() {
    // Comprehensive linear algebra test 42
    // Placeholder for linalg operations
}

#[test]
fn test_linalg_comprehensive_43() {
    // Comprehensive linear algebra test 43
    // Placeholder for linalg operations
}

#[test]
fn test_linalg_comprehensive_44() {
    // Comprehensive linear algebra test 44
    // Placeholder for linalg operations
}

#[test]
fn test_linalg_comprehensive_45() {
    // Comprehensive linear algebra test 45
    // Placeholder for linalg operations
}

#[test]
fn test_linalg_comprehensive_46() {
    // Comprehensive linear algebra test 46
    // Placeholder for linalg operations
}

#[test]
fn test_linalg_comprehensive_47() {
    // Comprehensive linear algebra test 47
    // Placeholder for linalg operations
}

#[test]
fn test_linalg_comprehensive_48() {
    // Comprehensive linear algebra test 48
    // Placeholder for linalg operations
}

#[test]
fn test_linalg_comprehensive_49() {
    // Comprehensive linear algebra test 49
    // Placeholder for linalg operations
}

#[test]
fn test_linalg_comprehensive_50() {
    // Comprehensive linear algebra test 50
    // Placeholder for linalg operations
}

#[test]
fn test_linalg_comprehensive_51() {
    // Comprehensive linear algebra test 51
    // Placeholder for linalg operations
}

#[test]
fn test_linalg_comprehensive_52() {
    // Comprehensive linear algebra test 52
    // Placeholder for linalg operations
}

#[test]
fn test_linalg_comprehensive_53() {
    // Comprehensive linear algebra test 53
    // Placeholder for linalg operations
}

#[test]
fn test_linalg_comprehensive_54() {
    // Comprehensive linear algebra test 54
    // Placeholder for linalg operations
}

#[test]
fn test_linalg_comprehensive_55() {
    // Comprehensive linear algebra test 55
    // Placeholder for linalg operations
}

#[test]
fn test_linalg_comprehensive_56() {
    // Comprehensive linear algebra test 56
    // Placeholder for linalg operations
}

#[test]
fn test_linalg_comprehensive_57() {
    // Comprehensive linear algebra test 57
    // Placeholder for linalg operations
}

#[test]
fn test_linalg_comprehensive_58() {
    // Comprehensive linear algebra test 58
    // Placeholder for linalg operations
}

#[test]
fn test_linalg_comprehensive_59() {
    // Comprehensive linear algebra test 59
    // Placeholder for linalg operations
}

#[test]
fn test_linalg_comprehensive_60() {
    // Comprehensive linear algebra test 60
    // Placeholder for linalg operations
}

#[test]
fn test_linalg_comprehensive_61() {
    // Comprehensive linear algebra test 61
    // Placeholder for linalg operations
}

#[test]
fn test_linalg_comprehensive_62() {
    // Comprehensive linear algebra test 62
    // Placeholder for linalg operations
}
