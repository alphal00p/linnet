use bitvec::vec::BitVec;
use bitvec_find::UnionFindBitFilter;

use super::*;

#[test]
fn test_union_find_basic() {
    // Use the public API only.
    let base_data = vec!['a', 'b', 'c', 'd', 'e'];
    let associated: Vec<u32> = base_data.iter().map(|&c| c as u32).collect();
    let mut uf = UnionFind::new(base_data, associated);

    // Initially, each element should be its own set.
    for i in 0..uf.elements.len() {
        let idx = ParentPointer(i);
        assert_eq!(uf.find(idx), idx);
        assert_eq!(*uf.find_data(idx), uf.elements[i] as u32);
    }

    // Union elements 0 and 1 (merging their associated data by summing the ASCII codes).
    let rep_ab = uf.union(ParentPointer(0), ParentPointer(1), |a, b| a + b);
    assert_eq!(uf.find(ParentPointer(0)), uf.find(ParentPointer(1)));
    assert_eq!(*uf.find_data(ParentPointer(0)), 97 + 98); // 195
    assert_eq!(rep_ab, uf.find(ParentPointer(0)));

    // Union elements 2 and 3.
    let rep_cd = uf.union(ParentPointer(2), ParentPointer(3), |a, b| a + b);
    assert_eq!(*uf.find_data(ParentPointer(2)), 99 + 100); // 199

    // Union the sets represented by rep_ab and rep_cd.
    let rep_abcd = uf.union(rep_ab, rep_cd, |a, b| a + b);
    assert_eq!(*uf.find_data(ParentPointer(0)), 195 + 199); // 394

    // Union an element with itself should not change the representative.
    let rep_same = uf.union(ParentPointer(0), ParentPointer(1), |a, b| a + b);
    assert_eq!(rep_same, uf.find(ParentPointer(0)));

    // Element 'e' (index 4) should remain in its own set.
    assert_ne!(uf.find(ParentPointer(0)), uf.find(ParentPointer(4)));
}

#[test]
fn test_union_find_bit_filter() {
    // Use the public API only.
    let elements = vec![10, 20, 30, 40, 50];
    let mut uf = UnionFindBitFilter::new(elements);

    // Use the public len() method.
    let n = uf.len();

    // Union sets for indices 0 and 1.
    let rep_0 = uf.union(ParentPointer(0), ParentPointer(1));
    {
        let filter = uf.find_filter(rep_0).clone();
        let mut expected: BitVec = BitVec::repeat(false, n);
        expected.set(0, true);
        expected.set(1, true);
        assert_eq!(filter, expected);
    }

    // Union sets for indices 2 and 3.
    let rep_2 = uf.union(ParentPointer(2), ParentPointer(3));
    {
        let filter = uf.find_filter(rep_2);
        let mut expected: BitVec = BitVec::repeat(false, n);
        expected.set(2, true);
        expected.set(3, true);
        assert_eq!(filter, &expected);
    }

    // Union the sets represented by rep_0 and rep_2.
    let rep_0_2 = uf.union(rep_0, rep_2);
    {
        let filter = uf.find_filter(rep_0_2);
        let mut expected: BitVec = BitVec::repeat(false, n);
        expected.set(0, true);
        expected.set(1, true);
        expected.set(2, true);
        expected.set(3, true);
        assert_eq!(filter, &expected);
    }

    // Element at index 4 should remain in its own set.
    {
        let filter = uf.find_filter(ParentPointer(4));
        let mut expected: BitVec = BitVec::repeat(false, n);
        expected.set(4, true);
        assert_eq!(filter, &expected);
    }

    // Test retrieval of base elements in a set.
    let elems_in_set = uf.elements_in_set(rep_0_2);
    // The set should contain 10, 20, 30, and 40.
    assert_eq!(elems_in_set.len(), 4);
    for &expected in &[10, 20, 30, 40] {
        assert!(elems_in_set.iter().any(|&&x| x == expected));
    }
}

#[test]
fn test_singleton_sets() {
    let uf = UnionFind::new(vec![10, 20, 30], vec!['a', 'b', 'c']);
    // Each element should be its own root.
    assert_eq!(uf.find(ParentPointer(0)), ParentPointer(0));
    assert_eq!(uf.find(ParentPointer(1)), ParentPointer(1));
    assert_eq!(uf.find(ParentPointer(2)), ParentPointer(2));

    // The associated data should match exactly.
    assert_eq!(*uf.find_data(ParentPointer(0)), 'a');
    assert_eq!(*uf.find_data(ParentPointer(1)), 'b');
    assert_eq!(*uf.find_data(ParentPointer(2)), 'c');
}

#[test]
fn test_union_basic() {
    let mut uf = UnionFind::new(vec![1, 2, 3, 4], vec![100, 200, 300, 400]);

    // Initially, all are separate
    for i in 0..4 {
        assert_eq!(uf.find(ParentPointer(i)), ParentPointer(i));
    }

    // Union of 0 and 1
    let r1 = uf.union(ParentPointer(0), ParentPointer(1), |x, y| x + y);
    // Root of 0 and 1 must now be the same
    assert_eq!(uf.find(ParentPointer(0)), uf.find(ParentPointer(1)));
    // That root must be `r1`
    assert_eq!(uf.find(ParentPointer(0)), r1);

    // The data in that root must be merged
    assert_eq!(*uf.find_data(r1), 300); // 100 + 200

    // Union of 2 and 3
    let r2 = uf.union(ParentPointer(2), ParentPointer(3), |x, y| x + y);
    assert_eq!(uf.find(ParentPointer(2)), uf.find(ParentPointer(3)));
    assert_eq!(*uf.find_data(r2), 700); // 300 + 400
}

#[test]
fn test_union_chain() {
    let mut uf = UnionFind::new(
        vec![
            "a".to_owned(),
            "b".to_owned(),
            "c".to_owned(),
            "d".to_owned(),
        ],
        vec![
            "DataA".to_owned(),
            "DataB".to_owned(),
            "DataC".to_owned(),
            "DataD".to_owned(),
        ],
    );

    // 0 U 1
    uf.union(ParentPointer(0), ParentPointer(1), |a, b| {
        format!("({}+{})", a, b)
    });
    // 2 U 3
    uf.union(ParentPointer(2), ParentPointer(3), |a, b| {
        format!("({}+{})", a, b)
    });
    // (0,1) U (2,3)
    let combined_root = uf.union(ParentPointer(0), ParentPointer(2), |a, b| {
        format!("({}|{})", a, b)
    });

    // The final data is some combination of DataA+DataB and DataC+DataD
    // e.g. "((DataA+DataB)|(DataC+DataD))"
    assert_eq!(
        *uf.find_data(combined_root),
        "((DataA+DataB)|(DataC+DataD))"
    );

    // All members now share the same root
    for i in 0..4 {
        assert_eq!(uf.find(ParentPointer(i)), combined_root);
    }
}

#[test]
fn test_union_noop_when_same_set() {
    let mut uf = UnionFind::new(vec![0, 1], vec![10, 20]);
    // Union of 0 with 1
    let first_root = uf.union(ParentPointer(0), ParentPointer(1), |a, b| a + b);

    // Union the same set again
    let second_root = uf.union(ParentPointer(0), ParentPointer(1), |a, b| a + b + 1000);

    // Because they were already in one set, no further merge occurs;
    // the root should not have changed in a meaningful way.
    assert_eq!(first_root, second_root);

    // The data should be 30 (10 + 20), not 1030, because no second merge should have happened
    assert_eq!(*uf.find_data(second_root), 30);
}

#[test]
fn test_swap_removal_correctness() {
    // This test checks that the swap removal logic is correct.
    let mut uf = UnionFind::new(
        vec![1, 2, 3],
        vec!["X".to_owned(), "Y".to_owned(), "Z".to_owned()],
    );
    // Merge 0 and 1
    let root_01 = uf.union(ParentPointer(0), ParentPointer(1), |x, y| {
        format!("{}{}", x, y)
    });
    // This will remove one slot from the associated data.
    // Then let's also union with 2
    let root_final = uf.union(root_01, ParentPointer(2), |x, y| format!("({}+{})", x, y));

    // All elements should be in the same set
    let r0 = uf.find(ParentPointer(0));
    let r1 = uf.find(ParentPointer(1));
    let r2 = uf.find(ParentPointer(2));

    assert_eq!(r0, root_final);
    assert_eq!(r1, root_final);
    assert_eq!(r2, root_final);

    // The final data depends on the merges, but we expect something like "(XY+Z)".
    assert_eq!(*uf.find_data(r0), "(XY+Z)");
}
