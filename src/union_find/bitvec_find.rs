use std::ops::{Index, IndexMut};

use bitvec::vec::BitVec;

use super::{ParentPointer, UnionFind};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LightIndex(pub usize);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HeavyIndex(pub usize);

pub type HLindex = HeavyLight<HeavyIndex, LightIndex>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HeavyLight<H, L> {
    Light(L),
    Heavy(H),
}

pub struct BitFilterData<U> {
    pub data: Option<U>,
    pub filter: BitVec,
}

pub struct UnionFindBitFilter<T, H, L> {
    pub inner: UnionFind<T, HLindex>,
    pub heavy_data: Vec<BitFilterData<H>>,
    pub light_data: Vec<BitFilterData<L>>,
}
impl<T, H, L> UnionFindBitFilter<T, H, L> {
    pub fn new_heavy(elements: Vec<T>, data: Vec<H>) -> Self {
        let n = elements.len();
        assert_eq!(n, data.len());

        let light_data = vec![];
        let mut heavy_data = vec![];
        let mut pointers = Vec::with_capacity(n);

        for (i, d) in data.into_iter().enumerate() {
            let mut filter: BitVec = BitVec::repeat(false, n);
            filter.set(i, true);
            let index = HeavyLight::Heavy(HeavyIndex(i));
            pointers.push(index);
            heavy_data.push(BitFilterData {
                data: Some(d),
                filter,
            });
        }

        let inner = UnionFind::new(elements, pointers);

        Self {
            inner,
            heavy_data,
            light_data,
        }
    }

    pub fn new_light(elements: Vec<T>, data: Vec<L>) -> Self {
        let n = elements.len();
        assert_eq!(n, data.len());

        let mut light_data = vec![];
        let heavy_data = vec![];
        let mut pointers = Vec::with_capacity(n);

        for (i, d) in data.into_iter().enumerate() {
            let mut filter: BitVec = BitVec::repeat(false, n);
            filter.set(i, true);
            let index = HeavyLight::Light(LightIndex(i));
            pointers.push(index);
            light_data.push(BitFilterData {
                data: Some(d),
                filter,
            });
        }

        let inner = UnionFind::new(elements, pointers);

        Self {
            inner,
            heavy_data,
            light_data,
        }
    }

    pub fn new(elements: Vec<T>, data_enum: Vec<HeavyLight<H, L>>) -> Self {
        let n = elements.len();
        assert_eq!(n, data_enum.len());

        let mut light_data = vec![];
        let mut heavy_data = vec![];
        let mut pointers = Vec::with_capacity(n);

        for (i, d) in data_enum.into_iter().enumerate() {
            let mut filter: BitVec = BitVec::repeat(false, n);
            filter.set(i, true);
            let index = match d {
                HeavyLight::Heavy(h) => {
                    let id = HeavyLight::Heavy(HeavyIndex(heavy_data.len()));
                    heavy_data.push(BitFilterData {
                        data: Some(h),
                        filter,
                    });
                    id
                }
                HeavyLight::Light(l) => {
                    let id = HeavyLight::Light(LightIndex(light_data.len()));
                    light_data.push(BitFilterData {
                        data: Some(l),
                        filter,
                    });
                    id
                }
            };

            pointers.push(index);
        }

        let inner = UnionFind::new(elements, pointers);

        Self {
            inner,
            heavy_data,
            light_data,
        }
    }

    /// Finds the representative of the set containing the element at ParentPointer x.
    pub fn find(&self, x: ParentPointer) -> ParentPointer {
        self.inner.find(x)
    }

    pub fn find_from_heavy(&self, h: HeavyIndex) -> ParentPointer {
        self.find(ParentPointer(self[&h].filter.iter_ones().next().unwrap()))
    }

    pub fn find_from_light(&self, l: LightIndex) -> ParentPointer {
        self.find(ParentPointer(self[&l].filter.iter_ones().next().unwrap()))
    }

    /// Returns a reference to the BitVec filter for the set containing the element at ParentPointer x.
    pub fn find_index(&self, x: ParentPointer) -> &HLindex {
        self.inner.find_data(x)
    }

    pub fn get(&self, hl_pointer: HLindex) -> HeavyLight<&H, &L> {
        match hl_pointer {
            HeavyLight::Heavy(h) => HeavyLight::Heavy(&self[h]),
            HeavyLight::Light(l) => HeavyLight::Light(&self[l]),
        }
    }

    pub fn find_data(&self, x: ParentPointer) -> HeavyLight<&H, &L> {
        let ptr = self.find_index(x);
        self.get(*ptr)
    }

    pub fn union<FH, FL>(
        &mut self,
        x: ParentPointer,
        y: ParentPointer,
        merge_h: FH,
        merge_l: FL,
    ) -> ParentPointer
    where
        FL: FnOnce(L, L) -> L,
        FH: FnOnce(H, H) -> H,
    {
        let mut loser_idx = None;

        let closure = |winner: HLindex, loser: HLindex| {
            match (winner, loser) {
                //-------------------------------------------------------
                // Case 1) Heavy vs. Heavy
                //-------------------------------------------------------
                (HeavyLight::Heavy(wi), HeavyLight::Heavy(li)) => {
                    let widx = wi.0;
                    let lidx = li.0;

                    // 1) Merge data
                    let wval = self.heavy_data[widx]
                        .data
                        .take()
                        .expect("Winner missing heavy data?");
                    let lval = self.heavy_data[lidx]
                        .data
                        .take()
                        .expect("Loser missing heavy data?");

                    self.heavy_data[widx].data = Some(merge_h(wval, lval));
                    if widx < lidx {
                        let (wslice, lslice) = self.heavy_data.split_at_mut(lidx);
                        wslice[widx].filter |= &lslice[0].filter;
                    } else {
                        let (lslice, wslice) = self.heavy_data.split_at_mut(widx);
                        wslice[0].filter |= &lslice[lidx].filter;
                    }
                }

                //-------------------------------------------------------
                // Case 2) Light vs. Light
                //-------------------------------------------------------
                (HeavyLight::Light(wi), HeavyLight::Light(li)) => {
                    let widx = wi.0;
                    let lidx = li.0;

                    // 1) Merge data
                    let wval = self.light_data[widx]
                        .data
                        .take()
                        .expect("Winner missing light data?");
                    let lval = self.light_data[lidx]
                        .data
                        .take()
                        .expect("Loser missing light data?");
                    self.light_data[widx].data = Some(merge_l(wval, lval));

                    // 2) Merge filters
                    if widx < lidx {
                        let (wslice, lslice) = self.light_data.split_at_mut(lidx);
                        wslice[widx].filter |= &lslice[0].filter;
                    } else {
                        let (lslice, wslice) = self.light_data.split_at_mut(widx);
                        wslice[0].filter |= &lslice[lidx].filter;
                    }
                }

                //-------------------------------------------------------
                // Case 3) Mismatch (can't unify a Heavy with a Light)
                //-------------------------------------------------------
                _ => panic!("Cannot unify a Heavy root with a Light root!"),
            }
            loser_idx = Some(loser);
            winner
        };
        let out = self.inner.union(x, y, closure);
        match loser_idx {
            Some(HeavyLight::Heavy(l)) => {
                if l.0 + 1 < self.heavy_data.len() {
                    self.heavy_data.swap_remove(l.0);
                    let set_index = self.inner.find_data_index(self.find_from_heavy(l));
                    self.inner[set_index] = HeavyLight::Heavy(l);
                } else {
                    self.heavy_data.pop();
                }
            }
            Some(HeavyLight::Light(l)) => {
                if l.0 + 1 < self.light_data.len() {
                    self.light_data.swap_remove(l.0);
                    let set_index = self.inner.find_data_index(self.find_from_light(l));
                    self.inner[set_index] = HeavyLight::Light(l);
                } else {
                    self.light_data.pop();
                }
            }
            _ => panic!("Cannot unify a Heavy root with a Light root!"),
        };
        out
    }
}

impl<T, H, L> Index<HeavyIndex> for UnionFindBitFilter<T, H, L> {
    type Output = H;
    fn index(&self, index: HeavyIndex) -> &Self::Output {
        self.heavy_data[index.0]
            .data
            .as_ref()
            .expect("missing heavy?")
    }
}

impl<T, H, L> IndexMut<HeavyIndex> for UnionFindBitFilter<T, H, L> {
    fn index_mut(&mut self, index: HeavyIndex) -> &mut Self::Output {
        self.heavy_data[index.0]
            .data
            .as_mut()
            .expect("missing heavy?")
    }
}

impl<T, H, L> Index<&HeavyIndex> for UnionFindBitFilter<T, H, L> {
    type Output = BitFilterData<H>;
    fn index(&self, index: &HeavyIndex) -> &Self::Output {
        &self.heavy_data[index.0]
    }
}

impl<T, H, L> IndexMut<&HeavyIndex> for UnionFindBitFilter<T, H, L> {
    fn index_mut(&mut self, index: &HeavyIndex) -> &mut Self::Output {
        &mut self.heavy_data[index.0]
    }
}

impl<T, H, L> Index<LightIndex> for UnionFindBitFilter<T, H, L> {
    type Output = L;
    fn index(&self, index: LightIndex) -> &Self::Output {
        self.light_data[index.0]
            .data
            .as_ref()
            .expect("missing light?")
    }
}

impl<T, H, L> IndexMut<LightIndex> for UnionFindBitFilter<T, H, L> {
    fn index_mut(&mut self, index: LightIndex) -> &mut Self::Output {
        self.light_data[index.0]
            .data
            .as_mut()
            .expect("missing light?")
    }
}

impl<T, H, L> Index<&LightIndex> for UnionFindBitFilter<T, H, L> {
    type Output = BitFilterData<L>;
    fn index(&self, index: &LightIndex) -> &Self::Output {
        &self.light_data[index.0]
    }
}

impl<T, H, L> IndexMut<&LightIndex> for UnionFindBitFilter<T, H, L> {
    fn index_mut(&mut self, index: &LightIndex) -> &mut Self::Output {
        &mut self.light_data[index.0]
    }
}
