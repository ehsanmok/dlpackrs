use thiserror::Error;

#[derive(Debug, Error)]
#[error("unsupported device: {0}")]
pub struct UnsupportedDeviceError(pub String);

#[derive(Debug, Error)]
#[error("unsupported data type code: {0}")]
pub struct UnsupportedDataTypeCode(pub String);
