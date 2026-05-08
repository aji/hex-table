use std::{
    fs::OpenOptions,
    io::{Seek, SeekFrom, Write},
};

fn main() {
    let mut f = OpenOptions::new()
        .create(true)
        .write(true)
        .read(true)
        .open("test.txt")
        .unwrap();
    f.seek(SeekFrom::Start(10)).unwrap();
    f.write_all(&[5]).unwrap();
}
