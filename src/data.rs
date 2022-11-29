use std::{fmt::Display, fs::File, io::Write};

/// Write out contents of `data` to CSV, with `label` in first column
pub fn write_csv<S, T>(file: &mut File, label: f64, data: &T) -> ()
where
  S: Display,
  for<'a> &'a T: IntoIterator<Item = &'a S>,
{
  file
    .write(format!("{label}").as_bytes())
    .expect("Unable to write to file");
  for val in data.into_iter() {
    file.write(b",").expect("Unable to write to file");
    file
      .write(format!("{val}").as_bytes())
      .expect("Unable to write to file");
  }
  file
    .write("\n".as_bytes())
    .expect("Unable to write to file");
}