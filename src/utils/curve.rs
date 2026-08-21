//! Wrap the curve used in the library.
//! 
//! We use the BN254 curve, which is a pairing-friendly curve.
//! 
//! We wrap the types and ops here to make the code more readable.
//! 
//! Note that we define the multiplication between a G1 element and a G2 element
//! as the pairing operation.
//!  
use core::convert::From;
use std::fs::File;
use std::io::Write;

use bincode;
use ark_bn254::{
    Bn254, Fr as Scalar, G1Affine, G1Projective, G2Affine, G2Projective,
};
use ark_ec::{
    pairing::{Pairing, PairingOutput},
    AffineRepr, CurveGroup, Group,
};
use ark_ff::{Field, PrimeField};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::{UniformRand, Zero as ArkZero};

use serde::ser::Serializer;
use serde::de::Deserializer;
use serde::{Serialize, Deserialize};

use crate::config::{DATA_DIR_PUBLIC, DATA_DIR_PRIVATE};

type Gt = PairingOutput<Bn254>;

pub use curv::arithmetic::Zero;
pub use std::ops::{Add, Mul, Neg, Sub, AddAssign};


#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub struct ZpElement { pub value: Scalar }

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub struct G1Element { pub value: G1Projective }

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub struct G2Element { pub value: G2Projective }

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub struct GtElement { pub value: Gt }

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub struct GtElementPack { pub value: Gt }




pub trait ConvertToZp: Copy + Clone + Send + Sync {
    fn to_zp(&self) -> ZpElement;
}

impl ConvertToZp for i64 {
    fn to_zp(&self) -> ZpElement {
        ZpElement ::from(*self)
    }
}
impl ConvertToZp for i128 {
    fn to_zp(&self) -> ZpElement {
        ZpElement ::from(*self)
    }
}
impl ConvertToZp for u64 {
    fn to_zp(&self) -> ZpElement {
        ZpElement ::from(*self)
    }
}

impl ConvertToZp for ZpElement {
    fn to_zp(&self) -> ZpElement {
        self.clone()
    }
}


pub trait Double {
    fn double(&self) -> Self;
}

fn serialize_canonical<T, S>(value: &T, serializer: S) -> Result<S::Ok, S::Error>
where
    T: CanonicalSerialize,
    S: Serializer,
{
    let mut bytes = Vec::new();
    value
        .serialize_compressed(&mut bytes)
        .map_err(serde::ser::Error::custom)?;
    serializer.serialize_bytes(&bytes)
}

fn deserialize_canonical<'de, T, D>(deserializer: D) -> Result<T, D::Error>
where
    T: CanonicalDeserialize,
    D: Deserializer<'de>,
{
    let bytes = Vec::<u8>::deserialize(deserializer)?;
    T::deserialize_compressed(bytes.as_slice()).map_err(serde::de::Error::custom)
}

impl Serialize for ZpElement {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serialize_canonical(&self.value, serializer)
    }
}

impl<'de> Deserialize<'de> for ZpElement{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>, 
    {
        Ok(ZpElement {
            value: deserialize_canonical(deserializer)?,
        })
    }
}

// #[derive(Serialize, Deserialize, Debug, PartialEq)]
// struct G1_BYTES_TYPE { bytes: [u8; 48] }

impl Serialize for G1Element {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {   
        serialize_canonical(&self.value.into_affine(), serializer)
    }
}

impl<'de> Deserialize<'de> for G1Element{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>, 
    {
        let value: G1Affine = deserialize_canonical(deserializer)?;
        Ok(G1Element { value: value.into_group() })
    }
}

impl Serialize for G2Element {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {   
        serialize_canonical(&self.value.into_affine(), serializer)
    }
}

impl<'de> Deserialize<'de> for G2Element{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>, 
    {
        let value: G2Affine = deserialize_canonical(deserializer)?;
        Ok(G2Element { value: value.into_group() })
    }
}

impl Serialize for GtElementPack {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {   
        serialize_canonical(&self.value, serializer)
    }
}

impl<'de> Deserialize<'de> for GtElementPack{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>, 
    {
        Ok(GtElementPack {
            value: deserialize_canonical(deserializer)?,
        })
    }
}

impl Serialize for GtElement {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {   
        let gt_pack = GtElementPack::from(*self);

        gt_pack.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for GtElement{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>, 
    {
        let gt_pack = GtElementPack::deserialize(deserializer)?;
        Ok(GtElement { value: gt_pack.value })
    }
}

trait ToFile {
    fn to_file(
        &self, file_name: String, public_data: bool
    ) -> std::io::Result<()>;
    fn from_file(
        file_name: String, public_data:bool
    ) -> std::io::Result<Self> 
    where
        Self: Sized;
}  

impl ToFile for GtElement {
    fn to_file(
        &self, file_name: String, public_data: bool
    ) -> std::io::Result<()> {
        
        let dir = if public_data {DATA_DIR_PUBLIC} else {DATA_DIR_PRIVATE};
        let mut file = File::create(format!("{}{}", dir, file_name))?;
        
        let encoded: Vec<u8> = bincode::serialize(&self).unwrap();
        file.write_all(&encoded)?;
        Ok(())
    }

    fn from_file(
        file_name: String, public_data: bool
    ) -> std::io::Result<Self> {
        
        let dir = if public_data {DATA_DIR_PUBLIC} else {DATA_DIR_PRIVATE};
        let file = File::open(format!("{}{}", dir, file_name))?;
        let decoded: Self = bincode::deserialize_from(file).unwrap();
        Ok(decoded)
    }
}

fn scalar_from_u128(value: u128) -> Scalar {
    Scalar::from_le_bytes_mod_order(&value.to_le_bytes())
}

impl ZpElement {
    pub fn rand() -> Self {
        ZpElement {
            value: Scalar::rand(&mut rand::thread_rng()),
        }
    }

    pub fn from_u8(bytes: &[u8; 32]) -> Self {
        ZpElement {
            value: Scalar::from_be_bytes_mod_order(bytes),
        }
    }

    pub fn pow(&self, exponent: u64) -> Self {
        let exponent_256bits:[u64;4] = [exponent.clone(), 0, 0, 0]; 
        ZpElement { value: self.value.pow(&exponent_256bits)}
    }

    pub fn inv(&self) -> Self {
        ZpElement { value: self.value.inverse().unwrap() }
    }
}

impl G1Element {
    pub fn generator() -> Self {
        G1Element { value: G1Projective::generator() }
    }
}

impl G2Element {
    pub fn generator() -> Self {
        G2Element { value: G2Projective::generator() }
    }
}

impl GtElement {
    pub fn generator() -> Self {
        let g1_gen = G1Affine::from(G1Projective::generator());
        let g2_gen = G2Affine::from(G2Projective::generator());
        GtElement { value: Bn254::pairing(g1_gen, g2_gen) }
    }
}

impl Zero for ZpElement {
    fn zero() -> Self {
        ZpElement { value: Scalar::zero() }
    }

    fn is_zero(&self) -> bool {
        return self.value == Scalar::zero();
    }
}

impl Zero for G1Element {
    fn zero() -> Self {
        G1Element { value: G1Projective::zero() }
    }

    fn is_zero(&self) -> bool {
        return self.value.is_zero();
    }
}

impl Zero for G2Element{
    fn zero() -> Self {
        G2Element { value: G2Projective::zero() }
    }

    fn is_zero(&self) -> bool {
        return self.value.is_zero();
    }
}

impl Zero for GtElement{
    fn zero() -> Self {
        GtElement { value: Gt::zero() }
    }

    fn is_zero(&self) -> bool {
        return self.value.is_zero();
    }
}


impl From<u64> for ZpElement {
    fn from(input_integer: u64) -> Self {
        ZpElement { 
            value: Scalar::from(input_integer),
        }
    }
}

impl From<u128> for ZpElement {
    fn from(input_integer: u128) -> Self {
        ZpElement { value: scalar_from_u128(input_integer) }
    }
}

impl From<GtElement> for GtElementPack{
    fn from(input_gt: GtElement) -> Self {
        GtElementPack { value: input_gt.value.clone() }
    }
}

impl From<i64> for ZpElement {
    fn from(input_integer: i64) -> Self {
        let value = Scalar::from(input_integer.unsigned_abs());
        ZpElement { value: if input_integer < 0 { -value } else { value } }
    }
}


impl From<i128> for ZpElement {
    fn from(input_integer: i128) -> Self {
        let value = scalar_from_u128(input_integer.unsigned_abs());
        ZpElement { value: if input_integer < 0 { -value } else { value } }
    }
}

macro_rules! impl_from_integer_for_group {
    ($integer:ty) => {
        impl From<$integer> for G1Element {
            fn from(value: $integer) -> Self {
                G1Element::generator() * ZpElement::from(value)
            }
        }

        impl From<$integer> for G2Element {
            fn from(value: $integer) -> Self {
                G2Element::generator() * ZpElement::from(value)
            }
        }

        impl From<$integer> for GtElement {
            fn from(value: $integer) -> Self {
                GtElement::generator() * ZpElement::from(value)
            }
        }
    };
}

impl_from_integer_for_group!(u64);
impl_from_integer_for_group!(u128);
impl_from_integer_for_group!(i64);
impl_from_integer_for_group!(i128);


impl Add for ZpElement {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        ZpElement { value: self.value + &rhs.value }
    }
}

impl Add for G1Element {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        G1Element { value: self.value + &rhs.value }
    }
}

impl Add for G2Element {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        G2Element { value: self.value + &rhs.value }
    }
}

impl Add for GtElement {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        GtElement { value: self.value + &rhs.value }
    }
}

impl AddAssign for ZpElement {
    fn add_assign(&mut self, rhs: Self) {
        self.value += &rhs.value;
    }
}

impl AddAssign for G1Element {
    fn add_assign(&mut self, rhs: Self) {
        self.value += &rhs.value;
    }
}

impl AddAssign for G2Element {
    fn add_assign(&mut self, rhs: Self) {
        self.value += &rhs.value;
    }
}

impl AddAssign for GtElement {
    fn add_assign(&mut self, rhs: Self) {
        self.value += &rhs.value;
    }
}

impl Sub for ZpElement {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        ZpElement { value: self.value - &rhs.value }
    }
}

impl Neg for ZpElement {
    type Output = Self;

    fn neg(self) -> Self::Output {
        ZpElement { value: -self.value }
    }
}

impl Sub for G1Element {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        G1Element { value: self.value - &rhs.value }
    }
}

impl Neg for G1Element {
    type Output = Self;

    fn neg(self) -> Self::Output {
        G1Element { value: -self.value }
    }
}

impl Sub for G2Element {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        G2Element { value: self.value - &rhs.value }
    }
}

impl Neg for G2Element {
    type Output = Self;

    fn neg(self) -> Self::Output {
        G2Element { value: -self.value }
    }
}

impl Sub for GtElement{
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        GtElement { value: self.value - &rhs.value }
    }
}

impl Neg for GtElement {
    type Output = Self;

    fn neg(self) -> Self::Output {
        GtElement { value: -self.value }
    }
}

impl Mul for ZpElement {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        ZpElement { value: self.value * &rhs.value }
    }

}

impl Mul<G1Element> for ZpElement {
    type Output = G1Element;

    fn mul(self, rhs: G1Element) -> Self::Output {
        G1Element { value: rhs.value * self.value }
    }
}

impl Mul<G2Element> for ZpElement {
    type Output = G2Element;

    fn mul(self, rhs: G2Element) -> Self::Output {
        G2Element { value: rhs.value * self.value}
    }
}

impl Mul<GtElement> for ZpElement {
    type Output = GtElement;

    fn mul(self, rhs: GtElement) -> Self::Output {
        GtElement { value: rhs.value * self.value}
    }
}

impl Mul<ZpElement> for u64 {
    type Output = ZpElement;

    fn mul(self, rhs: ZpElement) -> Self::Output {
        ZpElement { 
            value: Scalar::from(self) * &rhs.value 
        }
    }
}

impl Mul<G1Element> for u64 {
    type Output = G1Element;

    fn mul(self, rhs: G1Element) -> Self::Output {
        G1Element { 
            value: rhs.value * Scalar::from(self)
        }
    }
}

impl Mul<G2Element> for u64 {
    type Output = G2Element;

    fn mul(self, rhs: G2Element) -> Self::Output {
        G2Element { 
            value: rhs.value * Scalar::from(self)
        }
    }
}

impl Mul<GtElement> for u64 {
    type Output = GtElement;

    fn mul(self, rhs: GtElement) -> Self::Output {
        GtElement { 
            value: rhs.value * Scalar::from(self)
        }
    }
}

impl Mul<ZpElement> for i64 {
    type Output = ZpElement;

    fn mul(self, rhs: ZpElement) -> Self::Output {
        ZpElement { 
            value: ZpElement::from(self).value * &rhs.value 
        }
    }
}

impl Mul<G1Element> for i64 {
    type Output = G1Element;

    fn mul(self, rhs: G1Element) -> Self::Output {
        G1Element { 
            value: rhs.value * ZpElement::from(self).value
        }
    }
}

impl Mul<G2Element> for i64 {
    type Output = G2Element;

    fn mul(self, rhs: G2Element) -> Self::Output {
        G2Element { 
            value: rhs.value * ZpElement::from(self).value
        }
    }
}

impl Mul<GtElement> for i64 {
    type Output = GtElement;

    fn mul(self, rhs: GtElement) -> Self::Output {
        GtElement { 
            value: rhs.value * ZpElement::from(self).value
        }
    }
}


impl Mul<ZpElement> for i128 {
    type Output = ZpElement;

    fn mul(self, rhs: ZpElement) -> Self::Output {
        ZpElement { 
            value: ZpElement::from(self).value * &rhs.value 
        }
    }
}

impl Mul<G1Element> for i128 {
    type Output = G1Element;

    fn mul(self, rhs: G1Element) -> Self::Output {
        G1Element { 
            value: rhs.value * ZpElement::from(self).value
        }
    }
}

impl Mul<G2Element> for i128 {
    type Output = G2Element;

    fn mul(self, rhs: G2Element) -> Self::Output {
        G2Element { 
            value: rhs.value * ZpElement::from(self).value
        }
    }
}

impl Mul<GtElement> for i128 {
    type Output = GtElement;

    fn mul(self, rhs: GtElement) -> Self::Output {
        GtElement { 
            value: rhs.value * ZpElement::from(self).value
        }
    }
}

impl Mul<u64> for ZpElement {
    type Output = ZpElement;

    fn mul(self, rhs: u64) -> Self::Output {
        ZpElement { 
            value: self.value * &Scalar::from(rhs) 
        }
    }
}

impl Mul<u64> for G1Element{
    type Output = G1Element;

    fn mul(self, rhs: u64) -> Self::Output {
        G1Element { 
            value: self.value * &Scalar::from(rhs) 
        }
    }
}

impl Mul<u64> for G2Element{
    type Output = G2Element;

    fn mul(self, rhs: u64) -> Self::Output {
        G2Element { 
            value: self.value * &Scalar::from(rhs) 
        }
    }
}

impl Mul<u64> for GtElement{
    type Output = GtElement;

    fn mul(self, rhs: u64) -> Self::Output {
        GtElement { 
            value: self.value * &Scalar::from(rhs) 
        }
    }
}

impl Mul<i64> for ZpElement {
    type Output = ZpElement;

    fn mul(self, rhs: i64) -> Self::Output {
        ZpElement { 
            value: self.value * ZpElement::from(rhs).value
        }
    }
}

impl Mul<i64> for G1Element{
    type Output = G1Element;

    fn mul(self, rhs: i64) -> Self::Output {
        G1Element { 
            value: self.value * ZpElement::from(rhs).value 
        }
    }
}

impl Mul<i64> for G2Element{
    type Output = G2Element;

    fn mul(self, rhs: i64) -> Self::Output {
        G2Element { 
            value: self.value * ZpElement::from(rhs).value 
        }
    }
}

impl Mul<i64> for GtElement{
    type Output = GtElement;

    fn mul(self, rhs: i64) -> Self::Output {
        GtElement { 
            value: self.value * ZpElement::from(rhs).value 
        }
    }
}

impl Mul<i128> for ZpElement {
    type Output = ZpElement;

    fn mul(self, rhs: i128) -> Self::Output {
        ZpElement { 
            value: self.value * ZpElement::from(rhs).value 
        }
    }
}

impl Mul<i128> for G1Element{
    type Output = G1Element;

    fn mul(self, rhs: i128) -> Self::Output {
        G1Element { 
            value: self.value * ZpElement::from(rhs).value 
        }
    }
}

impl Mul<i128> for G2Element{
    type Output = G2Element;

    fn mul(self, rhs: i128) -> Self::Output {
        G2Element { 
            value: self.value * ZpElement::from(rhs).value 
        }
    }
}

impl Mul<i128> for GtElement{
    type Output = GtElement;

    fn mul(self, rhs: i128) -> Self::Output {
        GtElement { 
            value: self.value * ZpElement::from(rhs).value 
        }
    }
}

impl Mul<ZpElement> for G1Element {
    type Output = G1Element;

    fn mul(self, rhs: ZpElement) -> Self::Output {
        G1Element { value: self.value * rhs.value }
    }
}

impl Mul<ZpElement> for G2Element {
    type Output = G2Element;

    fn mul(self, rhs: ZpElement) -> Self::Output {
        G2Element { value: self.value * rhs.value }
    }
}

impl Mul<ZpElement> for GtElement {
    type Output = GtElement;

    fn mul(self, rhs: ZpElement) -> Self::Output {
        GtElement { value: self.value * rhs.value }
    }
}

impl Mul<G2Element> for G1Element {
    type Output = GtElement;

    fn mul(self, rhs: G2Element) -> Self::Output {
        let lfs_affine = G1Affine::from(self.value);
        let rhs_affine = G2Affine::from(rhs.value);
        GtElement { value: Bn254::pairing(lfs_affine, rhs_affine) }
    }
}

impl Mul<G1Element> for G2Element {
    type Output = GtElement;

    fn mul(self, rhs: G1Element) -> Self::Output {
        let lfs_affine = G2Affine::from(self.value);
        let rhs_affine = G1Affine::from(rhs.value);
        GtElement { value: Bn254::pairing(rhs_affine, lfs_affine) }
    }
}

impl Double for ZpElement{
    fn double(&self) -> Self {
        ZpElement { value: self.value.double() }
    }
}

impl Double for G1Element{
    fn double(&self) -> Self {
        G1Element { value: self.value.double() }
    }
}

impl Double for G2Element{
    fn double(&self) -> Self {
        G2Element { value: self.value.double() }
    }
}

impl Double for GtElement{
    fn double(&self) -> Self {
        GtElement { value: self.value.double() }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_curve() {
        let scalar_zero = ZpElement::zero();
        let g1_zero = G1Element::zero();
        let g2_zero = G2Element::zero();
        let gt_zero = GtElement::zero();

        assert_eq!(scalar_zero, ZpElement::from(0 as u64));
        assert_eq!(g1_zero, G1Element::from(0 as u64));
        assert_eq!(g2_zero, G2Element::from(0 as u64));
        assert_eq!(gt_zero, GtElement::from(0 as u64));

        let g1_value = G1Element::from(-2 as i64);
        let g2_value = G2Element::from(-3 as i64);
        let gt_value = GtElement::from(6 as i64);

        assert_eq!(g1_value * g2_value, gt_value);

        let g1_value = G1Element::from(-2 as i128);
        let g2_value = G2Element::from(-3 as i128);
        let gt_value = GtElement::from(6 as i128);

        assert_eq!(g1_value * g2_value, gt_value);

        let mut g1_mut = G1Element::from(-3 as i64);
        let mut g2_mut = G2Element::from(-4 as i64);
        let mut gt_mut = GtElement::from(5 as i64);

        g1_mut += G1Element::generator();
        g2_mut += G2Element::generator();
        gt_mut += GtElement::generator();

        assert_eq!(g1_mut, g1_value);
        assert_eq!(g2_mut, g2_value);
        assert_eq!(gt_mut, gt_value);

        let zp_value = ZpElement::from(2 as i64);
        let zp_exp = zp_value.pow(3);

        assert_eq!(zp_exp, ZpElement::from(8 as i64));

        assert_eq!(scalar_zero + scalar_zero, ZpElement::zero());
        assert_eq!(scalar_zero - scalar_zero, ZpElement::zero());
        assert_eq!(- scalar_zero - scalar_zero, ZpElement::zero());
        assert_eq!(- scalar_zero * scalar_zero, ZpElement::zero());
        assert_eq!(1 as u64 * scalar_zero, scalar_zero);
        assert_eq!(scalar_zero * 1 as u64, scalar_zero);
        assert_eq!(scalar_zero.double(), scalar_zero);

        assert_eq!(g1_value + g1_zero, g1_value);
        assert_eq!(g1_value - g1_zero, g1_value);
        assert_eq!(- g1_value - g1_zero, - g1_value);
        assert_eq!(- g1_value * scalar_zero, g1_zero);
        assert_eq!(scalar_zero * g1_value, g1_zero);
        assert_eq!(1 as u64 * g1_value, g1_value);
        assert_eq!(g1_value * 1 as u64, g1_value);
        assert_eq!(g1_value.double(), G1Element::from(-4 as i64));

        assert_eq!(g2_value + g2_zero, g2_value);
        assert_eq!(g2_value - g2_zero, g2_value);
        assert_eq!(- g2_value - g2_zero, - g2_value);
        assert_eq!(- g2_value * scalar_zero, g2_zero);
        assert_eq!(scalar_zero * g2_value, g2_zero);
        assert_eq!(1 as u64 * g2_value, g2_value);
        assert_eq!(g2_value * 1 as u64, g2_value);
        assert_eq!(g2_value.double(), G2Element::from(-6 as i64));

        assert_eq!(gt_value + gt_zero, gt_value);
        assert_eq!(gt_value - gt_zero, gt_value);
        assert_eq!(- gt_value - gt_zero, - gt_value);
        assert_eq!(- gt_value * scalar_zero, gt_zero);
        assert_eq!(scalar_zero * gt_value, gt_zero);
        assert_eq!(g1_zero * g2_zero, gt_zero);
        assert_eq!(1 as u64 * gt_value, gt_value);
        assert_eq!(gt_value * 1 as u64, gt_value);
        assert_eq!(gt_value.double(), GtElement::from(12 as i64));

        let serialized_data = 
        (zp_value, g1_value, g2_value, gt_value);
        let zp_ser = bincode::serialize(&zp_value).unwrap();
        let g1_ser = bincode::serialize(&g1_value).unwrap();
        let zp_de: ZpElement = bincode::deserialize(&zp_ser[..]).unwrap();
        let g1_de: G1Element = bincode::deserialize(&g1_ser[..]).unwrap();

        assert_eq!(zp_value, zp_de);
        assert_eq!(g1_value, g1_de);
        let encoded: Vec<u8> = bincode::serialize(&serialized_data).unwrap();
        
        let gt_value_pack = GtElementPack::from(gt_value);
        let gt_ser_pack = bincode::serialize(&gt_value_pack).unwrap();
        let gt_de_pack: GtElementPack = bincode::deserialize(
            &gt_ser_pack[..]).unwrap();
        assert_eq!(gt_value_pack, gt_de_pack);

        let gt_ser = bincode::serialize(&gt_value).unwrap();
        let gt_de: GtElement = bincode::deserialize(&gt_ser[..]).unwrap();
        assert_eq!(gt_value, gt_de);

        let decoded: (ZpElement, G1Element, G2Element, GtElement) = 
            bincode::deserialize(&encoded[..]).unwrap();
        assert_eq!(serialized_data, decoded);

        println!("Encoded: {:?}", serialized_data);
        println!("Decoded: {:?}", decoded);

    }

}
