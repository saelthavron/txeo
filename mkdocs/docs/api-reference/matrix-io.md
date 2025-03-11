# MatrixIO

`txeo::MatrixIO` is a class designed for convenient reading and writing of matrix data from and to text files. It supports flexible formatting options including custom separators and floating-point precision control.

## Constructors

### Initialization with file path and column separator

```cpp
explicit MatrixIO(const std::filesystem::path &path, char separator = ',');
```

Constructs a `MatrixIO` object associated with a specified file path and optional separator.

**Example:**

```cpp
txeo::MatrixIO io("data.csv");
```

## Member Functions

### read_text_file

```cpp
template <typename T>
txeo::Matrix<T> read_text_file(bool has_header = false) const;
```

Reads matrix data from a file into a `Matrix<T>`. Optionally skips the first line if it is a header.

**Example:**

```cpp
txeo::MatrixIO io("data.csv");
auto matrix = io.read_text_file<float>(true);
```

### write_text_file

```cpp
template <typename T>
void write_text_file(const txeo::Matrix<T> &matrix) const;
```

Writes a matrix to a file using the defined separator.

**Example:**

```cpp
txeo::Matrix<int> data(txeo::TensorShape({2, 3}), {1, 2, 3, 4, 5, 6});
txeo::MatrixIO io("output.csv");
io.write_text_file(data);
```

### write_text_file (with precision)

```cpp
template <typename T>
  requires(std::is_floating_point_v<T>)
void write_text_file(const txeo::Matrix<T> &matrix, size_t precision) const;
```

Writes a floating-point matrix to a file with a specified number of decimal places.

**Example:**

```cpp
txeo::Matrix<double> values(txeo::TensorShape({1, 3}), {1.2345, 2.3456, 3.4567});
txeo::MatrixIO io("results.csv");
io.write_text_file(values, 2); // Output: 1.23,2.35,3.46
```

## Static Member Functions

### read_textfile

```cpp
template <typename T>
static txeo::Matrix<T> read_textfile(const std::filesystem::path &path, char separator = ',', bool has_header = false);
```

Convenience static function for reading matrix data from a file in a single call.

**Example:**

```cpp
auto data = txeo::MatrixIO::read_textfile<float>("input.tsv", '\t', true);
```

### write_textfile

```cpp
template <typename T>
static void write_textfile(const txeo::Matrix<T> &matrix, const std::filesystem::path &path, char separator = ',');
```

Convenience static function for writing matrix data to a file in a single call.

**Example:**

```cpp
txeo::Matrix<int> matrix(txeo::TensorShape({3, 2}), {1, 2, 3, 4, 5, 6});
txeo::MatrixIO::write_textfile(matrix, "matrix.csv");
```

### write_textfile (with precision)

```cpp
template <typename T>
  requires(std::is_floating_point_v<T>)
static void write_textfile(const txeo::Matrix<T> &matrix, size_t precision, const std::filesystem::path &path, char separator = ',');
```

Writes a floating-point matrix to a file with specified precision using a single call.

**Example:**

```cpp
txeo::Matrix<double> results(txeo::TensorShape({2, 2}), {0.000123, 4567.8, 9.1, 234.567});
txeo::MatrixIO::write_textfile(results, 3, "science.csv");
// Output: 0.000,4567.800,9.100,234.567
```

## Exceptions

All methods may throw:

### MatrixIOError

Exception class thrown on I/O errors.

```cpp
class MatrixIOError : public std::runtime_error;
```

---

For detailed API references, see individual method documentation at [txeo::MatrixIO](https://txeo-doc.netlify.app/classtxeo_1_1_matrix_i_o.html).
