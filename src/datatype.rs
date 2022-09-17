#![allow(non_upper_case_globals)]

use std::convert::TryFrom;

use crate::{
    errors::UnsupportedDataTypeCode,
    ffi::{
        DLDataType, DLDataTypeCode, DLDataTypeCode_kDLBfloat, DLDataTypeCode_kDLComplex,
        DLDataTypeCode_kDLFloat, DLDataTypeCode_kDLInt, DLDataTypeCode_kDLOpaqueHandle,
        DLDataTypeCode_kDLUInt,
    },
};

/// See [DLDataTypeCode](https://dmlc.github.io/dlpack/latest/c_api.html#_CPPv414DLDataTypeCode)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum DataTypeCode {
    Int = 0,
    UInt = 1,
    Float = 2,
    OpaqueHandle = 3,
    Bfloat = 4,
    Complex = 5,
}

impl From<DataTypeCode> for u8 {
    fn from(code: DataTypeCode) -> Self {
        match code {
            DataTypeCode::Int => 0,
            DataTypeCode::UInt => 1,
            DataTypeCode::Float => 2,
            DataTypeCode::OpaqueHandle => 3,
            DataTypeCode::Bfloat => 4,
            DataTypeCode::Complex => 5,
        }
    }
}

impl<'a> From<&'a DataTypeCode> for DLDataTypeCode {
    fn from(code: &'a DataTypeCode) -> Self {
        match code {
            DataTypeCode::Int => DLDataTypeCode_kDLInt,
            DataTypeCode::UInt => DLDataTypeCode_kDLUInt,
            DataTypeCode::Float => DLDataTypeCode_kDLFloat,
            DataTypeCode::OpaqueHandle => DLDataTypeCode_kDLOpaqueHandle,
            DataTypeCode::Bfloat => DLDataTypeCode_kDLBfloat,
            DataTypeCode::Complex => DLDataTypeCode_kDLComplex,
        }
    }
}

impl TryFrom<DLDataTypeCode> for DataTypeCode {
    type Error = UnsupportedDataTypeCode;
    fn try_from(code: DLDataTypeCode) -> Result<Self, Self::Error> {
        match code {
            DLDataTypeCode_kDLInt => Ok(DataTypeCode::Int),
            DLDataTypeCode_kDLUInt => Ok(DataTypeCode::UInt),
            DLDataTypeCode_kDLFloat => Ok(DataTypeCode::Float),
            DLDataTypeCode_kDLOpaqueHandle => Ok(DataTypeCode::OpaqueHandle),
            DLDataTypeCode_kDLBfloat => Ok(DataTypeCode::Bfloat),
            DLDataTypeCode_kDLComplex => Ok(DataTypeCode::Complex),
            _ => Err(UnsupportedDataTypeCode(code.to_string())),
        }
    }
}

/// DLPack DataType. See [DLDataType](https://dmlc.github.io/dlpack/latest/c_api.html#_CPPv410DLDataType)
///
/// ## Example
///
/// ```
/// use dlpackrs::DataType;
/// let i32 = DataType::i32();
/// println!("bits: {}, lanes: {}", i32.bits, i32.lanes);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(C)]
pub struct DataType {
    pub code: u8,
    pub bits: u8,
    pub lanes: u16,
}

impl From<DataType> for DLDataType {
    fn from(dtype: DataType) -> Self {
        Self {
            code: dtype.code as u8,
            bits: dtype.bits as u8,
            lanes: dtype.lanes as u16,
        }
    }
}

impl From<DLDataType> for DataType {
    fn from(dtype: DLDataType) -> Self {
        Self {
            code: dtype.code,
            bits: dtype.bits,
            lanes: dtype.lanes,
        }
    }
}

impl DataType {
    pub const fn new(code: u8, bits: u8, lanes: u16) -> DataType {
        DataType { code, bits, lanes }
    }

    pub const fn code(&self) -> usize {
        self.code as usize
    }

    pub const fn bits(&self) -> usize {
        self.bits as usize
    }

    pub const fn lanes(&self) -> usize {
        self.lanes as usize
    }

    /// For vectorized int type.
    pub fn int(bits: u8, lanes: u16) -> DataType {
        DataType::new(DataTypeCode::Int.into(), bits, lanes)
    }

    pub fn i8() -> DataType {
        Self::int(8, 1)
    }

    pub fn i16() -> DataType {
        Self::int(16, 1)
    }

    pub fn i32() -> DataType {
        Self::int(32, 1)
    }

    pub fn i64() -> DataType {
        Self::int(64, 1)
    }

    /// For vectorized uint type.
    pub fn uint(bits: u8, lanes: u16) -> DataType {
        DataType::new(DataTypeCode::UInt.into(), bits, lanes)
    }

    pub fn u8() -> DataType {
        Self::uint(8, 1)
    }

    pub fn u16() -> DataType {
        Self::uint(16, 1)
    }

    pub fn u32() -> DataType {
        Self::uint(32, 1)
    }

    pub fn u64() -> DataType {
        Self::uint(64, 1)
    }

    pub fn float(bits: u8, lanes: u16) -> DataType {
        DataType::new(DataTypeCode::Float.into(), bits, lanes)
    }

    pub fn f32() -> DataType {
        Self::float(32, 1)
    }

    pub fn f64() -> DataType {
        Self::float(64, 1)
    }

    /// Opaque handle type.
    pub fn opaque_handle(bits: u8, lanes: u16) -> DataType {
        DataType::new(DataTypeCode::OpaqueHandle.into(), bits, lanes)
    }

    /// BFloat type.
    pub fn bfloat(bits: u8, lanes: u16) -> DataType {
        DataType::new(DataTypeCode::Bfloat.into(), bits, lanes)
    }

    /// Mathematical Complex type.
    pub fn complex(bits: u8, lanes: u16) -> DataType {
        DataType::new(DataTypeCode::Complex.into(), bits, lanes)
    }
}
