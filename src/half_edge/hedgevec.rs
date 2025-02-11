use std::ops::{Index, IndexMut};

use serde::{Deserialize, Serialize};

use super::involution::{EdgeData, EdgeIndex, Hedge, HedgePair, Involution};

pub struct SmartHedgeVec<T> {
    pub(super) data: Vec<(HedgePair, T)>,
    pub(super) involution: Involution<EdgeIndex>,
}

impl<T> SmartHedgeVec<T> {
    pub fn inv(&self, hedge: Hedge) -> Hedge {
        self.involution.inv(hedge)
    }

    pub fn mapped_from_involution<E>(
        involution: &Involution<E>,
        f: &impl Fn(HedgePair, EdgeData<&E>) -> EdgeData<T>,
    ) -> Self {
        let mut data = Vec::new();

        let involution = involution.as_ref().map_full(|a, d| {
            let new_data = f(a, d);
            let edgeid = EdgeIndex(data.len());
            data.push((a, new_data.data));
            EdgeData::new(edgeid, new_data.orientation)
        });
        SmartHedgeVec { data, involution }
    }
}

impl<T> IntoIterator for SmartHedgeVec<T> {
    type Item = (EdgeIndex, HedgePair, T);
    type IntoIter = std::iter::Map<
        std::iter::Enumerate<std::vec::IntoIter<(HedgePair, T)>>,
        fn((usize, (HedgePair, T))) -> (EdgeIndex, HedgePair, T),
    >;
    fn into_iter(self) -> Self::IntoIter {
        self.data
            .into_iter()
            .enumerate()
            .map(|(u, t)| (EdgeIndex(u), t.0, t.1))
    }
}

impl<'a, T> IntoIterator for &'a SmartHedgeVec<T> {
    type Item = (EdgeIndex, HedgePair, &'a T);
    type IntoIter = std::iter::Map<
        std::iter::Enumerate<std::slice::Iter<'a, (HedgePair, T)>>,
        fn((usize, &(HedgePair, T))) -> (EdgeIndex, HedgePair, &T),
    >;
    fn into_iter(self) -> Self::IntoIter {
        self.data
            .iter()
            .enumerate()
            .map(|(u, t)| (EdgeIndex(u), t.0, &t.1))
    }
}

impl<T> IndexMut<EdgeIndex> for SmartHedgeVec<T> {
    fn index_mut(&mut self, index: EdgeIndex) -> &mut Self::Output {
        &mut self.data[index.0].1
    }
}
impl<T> Index<EdgeIndex> for SmartHedgeVec<T> {
    type Output = T;
    fn index(&self, index: EdgeIndex) -> &Self::Output {
        &self.data[index.0].1
    }
}

impl<T> Index<&EdgeIndex> for SmartHedgeVec<T> {
    type Output = (HedgePair, T);
    fn index(&self, index: &EdgeIndex) -> &Self::Output {
        &self.data[index.0]
    }
}

impl<T> Index<Hedge> for SmartHedgeVec<T> {
    type Output = T;
    fn index(&self, hedge: Hedge) -> &Self::Output {
        let eid = self.involution[hedge];
        &self[eid]
    }
}

impl<T> IndexMut<Hedge> for SmartHedgeVec<T> {
    fn index_mut(&mut self, hedge: Hedge) -> &mut Self::Output {
        let eid = self.involution[hedge];
        &mut self[eid]
    }
}

impl<T> Index<&Hedge> for SmartHedgeVec<T> {
    type Output = (HedgePair, T);
    fn index(&self, hedge: &Hedge) -> &Self::Output {
        let eid = self.involution[*hedge];
        &self[&eid]
    }
}

// Data stored once per edge (pair of half-edges or external edge)
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HedgeVec<T>(pub(super) Vec<T>);

impl<T> IntoIterator for HedgeVec<T> {
    type Item = (EdgeIndex, T);
    type IntoIter = std::iter::Map<
        std::iter::Enumerate<std::vec::IntoIter<T>>,
        fn((usize, T)) -> (EdgeIndex, T),
    >;
    fn into_iter(self) -> Self::IntoIter {
        self.0
            .into_iter()
            .enumerate()
            .map(|(u, t)| (EdgeIndex(u), t))
    }
}

impl<'a, T> IntoIterator for &'a HedgeVec<T> {
    type Item = (EdgeIndex, &'a T);
    type IntoIter = std::iter::Map<
        std::iter::Enumerate<std::slice::Iter<'a, T>>,
        fn((usize, &T)) -> (EdgeIndex, &T),
    >;
    fn into_iter(self) -> Self::IntoIter {
        self.0.iter().enumerate().map(|(u, t)| (EdgeIndex(u), t))
    }
}

impl<T> IndexMut<EdgeIndex> for HedgeVec<T> {
    fn index_mut(&mut self, index: EdgeIndex) -> &mut Self::Output {
        &mut self.0[index.0]
    }
}
impl<T> Index<EdgeIndex> for HedgeVec<T> {
    type Output = T;
    fn index(&self, index: EdgeIndex) -> &Self::Output {
        &self.0[index.0]
    }
}
