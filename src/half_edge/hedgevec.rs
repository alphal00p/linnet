use std::ops::{Index, IndexMut};

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

// Data stored once per edge (pair of half-edges or external edge)
pub struct HedgeVec<T>(pub(super) Vec<T>);

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
