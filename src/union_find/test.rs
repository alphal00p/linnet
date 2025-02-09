use super::*;
use crate::union_find::bitvec_find::*;

// Helper functions for testing
fn sum_merge(a: i32, b: i32) -> i32 {
    a + b
}
fn concat_merge(a: String, b: String) -> String {
    format!("{}{}", a, b)
}

#[test]
fn test_basic_union_find() {
    let elements = vec![1, 2, 3, 4, 5];
    let data = vec![10, 20, 30, 40, 50];
    let mut uf = UnionFind::new(elements, data);

    // Test initial state
    assert_eq!(*uf.find_data(ParentPointer(0)), 10);
    assert_eq!(*uf.find_data(ParentPointer(1)), 20);

    // Test union
    let _ = uf.union(ParentPointer(0), ParentPointer(1), sum_merge);
    assert_eq!(*uf.find_data(ParentPointer(0)), 30); // 10 + 20
    assert_eq!(uf.find(ParentPointer(0)), uf.find(ParentPointer(1)));
}

#[test]
fn test_path_compression() {
    let elements = vec![1, 2, 3, 4];
    let data = vec![10, 20, 30, 40];
    let mut uf = UnionFind::new(elements, data);

    // Create a chain: 3->2->1->0
    uf.union(ParentPointer(0), ParentPointer(1), sum_merge);
    uf.union(ParentPointer(1), ParentPointer(2), sum_merge);
    uf.union(ParentPointer(2), ParentPointer(3), sum_merge);

    // Find should compress the path
    let root = uf.find(ParentPointer(3));
    assert_eq!(uf.find(ParentPointer(2)), root);
    assert_eq!(uf.find(ParentPointer(1)), root);
}

#[test]
fn test_union_find_bit_filter() {
    // Test heavy elements
    let elements = vec![1, 2, 3];
    let heavy_data = vec![10, 20, 30];
    let mut uf = UnionFindBitFilter::new_heavy(elements, heavy_data);

    // Test initial state
    assert_eq!(uf[HeavyIndex(0)], 10);
    assert_eq!(uf[HeavyIndex(1)], 20);

    // Test union of heavy elements
    uf.union(ParentPointer(0), ParentPointer(1), sum_merge, concat_merge);
    assert_eq!(uf[HeavyIndex(0)], 30); // 10 + 20
}

#[test]
fn test_light_elements() {
    let elements = vec![1, 2, 3];
    let light_data = vec!["a".to_string(), "b".to_string(), "c".to_string()];
    let mut uf = UnionFindBitFilter::new_light(elements, light_data);

    // Test initial state
    assert_eq!(uf[LightIndex(0)], "a");
    assert_eq!(uf[LightIndex(1)], "b");

    // Test union of light elements
    uf.union(ParentPointer(0), ParentPointer(1), sum_merge, concat_merge);
    assert_eq!(uf[LightIndex(0)], "ab");
}

#[test]
#[should_panic(expected = "Cannot unify a Heavy root with a Light root!")]
fn test_heavy_light_union_panic() {
    let elements = vec![1, 2];
    let data_enum = vec![HeavyLight::Heavy(10), HeavyLight::Light("a".to_string())];
    let mut uf = UnionFindBitFilter::new(elements, data_enum);

    // This should panic
    uf.union(ParentPointer(0), ParentPointer(1), sum_merge, concat_merge);
}

#[test]
fn test_mixed_elements() {
    let elements = vec![1, 2, 3, 4];
    let data_enum = vec![
        HeavyLight::Heavy(10),
        HeavyLight::Heavy(20),
        HeavyLight::Light("a".to_string()),
        HeavyLight::Light("b".to_string()),
    ];
    let uf = UnionFindBitFilter::new(elements, data_enum);

    // Test correct type assignment
    match uf.get(*uf.find_index(ParentPointer(0))) {
        HeavyLight::Heavy(h) => assert_eq!(*h, 10),
        _ => panic!("Wrong type"),
    }

    match uf.get(*uf.find_index(ParentPointer(2))) {
        HeavyLight::Light(l) => assert_eq!(l, "a"),
        _ => panic!("Wrong type"),
    }
}

#[test]
fn test_bitvec_filter() {
    let elements = vec![1, 2, 3];
    let heavy_data = vec![10, 20, 30];
    let uf: UnionFindBitFilter<i32, i32, i8> = UnionFindBitFilter::new_heavy(elements, heavy_data);

    // Test initial filter state
    for i in 0..3 {
        let filter = &uf[&HeavyIndex(i)].filter;
        for j in 0..3 {
            assert_eq!(filter[j], i == j);
        }
    }
}

#[test]
fn test_find_from_heavy_light() {
    let elements = vec![1, 2, 3, 4];
    let data_enum = vec![
        HeavyLight::Heavy(10),
        HeavyLight::Heavy(20),
        HeavyLight::Light("a".to_string()),
        HeavyLight::Light("b".to_string()),
    ];
    let uf = UnionFindBitFilter::new(elements, data_enum);

    assert_eq!(uf.find_from_heavy(HeavyIndex(0)), ParentPointer(0));
    assert_eq!(uf.find_from_light(LightIndex(0)), ParentPointer(2));
}
