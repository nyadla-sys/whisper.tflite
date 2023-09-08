// automatically generated by the FlatBuffers compiler, do not modify


#ifndef FLATBUFFERS_GENERATED_ARRAYSTEST_MYGAME_EXAMPLE_H_
#define FLATBUFFERS_GENERATED_ARRAYSTEST_MYGAME_EXAMPLE_H_

#include "flatbuffers/flatbuffers.h"

// Ensure the included flatbuffers.h is the same version as when this file was
// generated, otherwise it may not be compatible.
static_assert(FLATBUFFERS_VERSION_MAJOR == 22 &&
              FLATBUFFERS_VERSION_MINOR == 10 &&
              FLATBUFFERS_VERSION_REVISION == 26,
             "Non-compatible flatbuffers version included");

namespace MyGame {
namespace Example {

struct NestedStruct;

struct ArrayStruct;

struct ArrayTable;
struct ArrayTableBuilder;
struct ArrayTableT;

bool operator==(const NestedStruct &lhs, const NestedStruct &rhs);
bool operator!=(const NestedStruct &lhs, const NestedStruct &rhs);
bool operator==(const ArrayStruct &lhs, const ArrayStruct &rhs);
bool operator!=(const ArrayStruct &lhs, const ArrayStruct &rhs);
bool operator==(const ArrayTableT &lhs, const ArrayTableT &rhs);
bool operator!=(const ArrayTableT &lhs, const ArrayTableT &rhs);

inline const flatbuffers::TypeTable *NestedStructTypeTable();

inline const flatbuffers::TypeTable *ArrayStructTypeTable();

inline const flatbuffers::TypeTable *ArrayTableTypeTable();

enum class TestEnum : int8_t {
  A = 0,
  B = 1,
  C = 2,
  MIN = A,
  MAX = C
};

inline const TestEnum (&EnumValuesTestEnum())[3] {
  static const TestEnum values[] = {
    TestEnum::A,
    TestEnum::B,
    TestEnum::C
  };
  return values;
}

inline const char * const *EnumNamesTestEnum() {
  static const char * const names[4] = {
    "A",
    "B",
    "C",
    nullptr
  };
  return names;
}

inline const char *EnumNameTestEnum(TestEnum e) {
  if (flatbuffers::IsOutRange(e, TestEnum::A, TestEnum::C)) return "";
  const size_t index = static_cast<size_t>(e);
  return EnumNamesTestEnum()[index];
}

FLATBUFFERS_MANUALLY_ALIGNED_STRUCT(8) NestedStruct FLATBUFFERS_FINAL_CLASS {
 private:
  int32_t a_[2];
  int8_t b_;
  int8_t c_[2];
  int8_t padding0__;  int32_t padding1__;
  int64_t d_[2];

 public:
  static const flatbuffers::TypeTable *MiniReflectTypeTable() {
    return NestedStructTypeTable();
  }
  NestedStruct()
      : a_(),
        b_(0),
        c_(),
        padding0__(0),
        padding1__(0),
        d_() {
    (void)padding0__;
    (void)padding1__;
  }
  NestedStruct(MyGame::Example::TestEnum _b)
      : a_(),
        b_(flatbuffers::EndianScalar(static_cast<int8_t>(_b))),
        c_(),
        padding0__(0),
        padding1__(0),
        d_() {
    (void)padding0__;
    (void)padding1__;
  }
  NestedStruct(flatbuffers::span<const int32_t, 2> _a, MyGame::Example::TestEnum _b, flatbuffers::span<const MyGame::Example::TestEnum, 2> _c, flatbuffers::span<const int64_t, 2> _d)
      : b_(flatbuffers::EndianScalar(static_cast<int8_t>(_b))),
        padding0__(0),
        padding1__(0) {
    flatbuffers::CastToArray(a_).CopyFromSpan(_a);
    flatbuffers::CastToArrayOfEnum<MyGame::Example::TestEnum>(c_).CopyFromSpan(_c);
    (void)padding0__;
    (void)padding1__;
    flatbuffers::CastToArray(d_).CopyFromSpan(_d);
  }
  const flatbuffers::Array<int32_t, 2> *a() const {
    return &flatbuffers::CastToArray(a_);
  }
  flatbuffers::Array<int32_t, 2> *mutable_a() {
    return &flatbuffers::CastToArray(a_);
  }
  MyGame::Example::TestEnum b() const {
    return static_cast<MyGame::Example::TestEnum>(flatbuffers::EndianScalar(b_));
  }
  void mutate_b(MyGame::Example::TestEnum _b) {
    flatbuffers::WriteScalar(&b_, static_cast<int8_t>(_b));
  }
  const flatbuffers::Array<MyGame::Example::TestEnum, 2> *c() const {
    return &flatbuffers::CastToArrayOfEnum<MyGame::Example::TestEnum>(c_);
  }
  flatbuffers::Array<MyGame::Example::TestEnum, 2> *mutable_c() {
    return &flatbuffers::CastToArrayOfEnum<MyGame::Example::TestEnum>(c_);
  }
  const flatbuffers::Array<int64_t, 2> *d() const {
    return &flatbuffers::CastToArray(d_);
  }
  flatbuffers::Array<int64_t, 2> *mutable_d() {
    return &flatbuffers::CastToArray(d_);
  }
};
FLATBUFFERS_STRUCT_END(NestedStruct, 32);

inline bool operator==(const NestedStruct &lhs, const NestedStruct &rhs) {
  return
      (lhs.a() == rhs.a()) &&
      (lhs.b() == rhs.b()) &&
      (lhs.c() == rhs.c()) &&
      (lhs.d() == rhs.d());
}

inline bool operator!=(const NestedStruct &lhs, const NestedStruct &rhs) {
    return !(lhs == rhs);
}


FLATBUFFERS_MANUALLY_ALIGNED_STRUCT(8) ArrayStruct FLATBUFFERS_FINAL_CLASS {
 private:
  float a_;
  int32_t b_[15];
  int8_t c_;
  int8_t padding0__;  int16_t padding1__;  int32_t padding2__;
  MyGame::Example::NestedStruct d_[2];
  int32_t e_;
  int32_t padding3__;
  int64_t f_[2];

 public:
  static const flatbuffers::TypeTable *MiniReflectTypeTable() {
    return ArrayStructTypeTable();
  }
  ArrayStruct()
      : a_(0),
        b_(),
        c_(0),
        padding0__(0),
        padding1__(0),
        padding2__(0),
        d_(),
        e_(0),
        padding3__(0),
        f_() {
    (void)padding0__;
    (void)padding1__;
    (void)padding2__;
    (void)padding3__;
  }
  ArrayStruct(float _a, int8_t _c, int32_t _e)
      : a_(flatbuffers::EndianScalar(_a)),
        b_(),
        c_(flatbuffers::EndianScalar(_c)),
        padding0__(0),
        padding1__(0),
        padding2__(0),
        d_(),
        e_(flatbuffers::EndianScalar(_e)),
        padding3__(0),
        f_() {
    (void)padding0__;
    (void)padding1__;
    (void)padding2__;
    (void)padding3__;
  }
  ArrayStruct(float _a, flatbuffers::span<const int32_t, 15> _b, int8_t _c, flatbuffers::span<const MyGame::Example::NestedStruct, 2> _d, int32_t _e, flatbuffers::span<const int64_t, 2> _f)
      : a_(flatbuffers::EndianScalar(_a)),
        c_(flatbuffers::EndianScalar(_c)),
        padding0__(0),
        padding1__(0),
        padding2__(0),
        e_(flatbuffers::EndianScalar(_e)),
        padding3__(0) {
    flatbuffers::CastToArray(b_).CopyFromSpan(_b);
    (void)padding0__;
    (void)padding1__;
    (void)padding2__;
    flatbuffers::CastToArray(d_).CopyFromSpan(_d);
    (void)padding3__;
    flatbuffers::CastToArray(f_).CopyFromSpan(_f);
  }
  float a() const {
    return flatbuffers::EndianScalar(a_);
  }
  void mutate_a(float _a) {
    flatbuffers::WriteScalar(&a_, _a);
  }
  const flatbuffers::Array<int32_t, 15> *b() const {
    return &flatbuffers::CastToArray(b_);
  }
  flatbuffers::Array<int32_t, 15> *mutable_b() {
    return &flatbuffers::CastToArray(b_);
  }
  int8_t c() const {
    return flatbuffers::EndianScalar(c_);
  }
  void mutate_c(int8_t _c) {
    flatbuffers::WriteScalar(&c_, _c);
  }
  const flatbuffers::Array<MyGame::Example::NestedStruct, 2> *d() const {
    return &flatbuffers::CastToArray(d_);
  }
  flatbuffers::Array<MyGame::Example::NestedStruct, 2> *mutable_d() {
    return &flatbuffers::CastToArray(d_);
  }
  int32_t e() const {
    return flatbuffers::EndianScalar(e_);
  }
  void mutate_e(int32_t _e) {
    flatbuffers::WriteScalar(&e_, _e);
  }
  const flatbuffers::Array<int64_t, 2> *f() const {
    return &flatbuffers::CastToArray(f_);
  }
  flatbuffers::Array<int64_t, 2> *mutable_f() {
    return &flatbuffers::CastToArray(f_);
  }
};
FLATBUFFERS_STRUCT_END(ArrayStruct, 160);

inline bool operator==(const ArrayStruct &lhs, const ArrayStruct &rhs) {
  return
      (lhs.a() == rhs.a()) &&
      (lhs.b() == rhs.b()) &&
      (lhs.c() == rhs.c()) &&
      (lhs.d() == rhs.d()) &&
      (lhs.e() == rhs.e()) &&
      (lhs.f() == rhs.f());
}

inline bool operator!=(const ArrayStruct &lhs, const ArrayStruct &rhs) {
    return !(lhs == rhs);
}


struct ArrayTableT : public flatbuffers::NativeTable {
  typedef ArrayTable TableType;
  flatbuffers::unique_ptr<MyGame::Example::ArrayStruct> a{};
  ArrayTableT() = default;
  ArrayTableT(const ArrayTableT &o);
  ArrayTableT(ArrayTableT&&) FLATBUFFERS_NOEXCEPT = default;
  ArrayTableT &operator=(ArrayTableT o) FLATBUFFERS_NOEXCEPT;
};

struct ArrayTable FLATBUFFERS_FINAL_CLASS : private flatbuffers::Table {
  typedef ArrayTableT NativeTableType;
  typedef ArrayTableBuilder Builder;
  static const flatbuffers::TypeTable *MiniReflectTypeTable() {
    return ArrayTableTypeTable();
  }
  enum FlatBuffersVTableOffset FLATBUFFERS_VTABLE_UNDERLYING_TYPE {
    VT_A = 4
  };
  const MyGame::Example::ArrayStruct *a() const {
    return GetStruct<const MyGame::Example::ArrayStruct *>(VT_A);
  }
  MyGame::Example::ArrayStruct *mutable_a() {
    return GetStruct<MyGame::Example::ArrayStruct *>(VT_A);
  }
  bool Verify(flatbuffers::Verifier &verifier) const {
    return VerifyTableStart(verifier) &&
           VerifyField<MyGame::Example::ArrayStruct>(verifier, VT_A, 8) &&
           verifier.EndTable();
  }
  ArrayTableT *UnPack(const flatbuffers::resolver_function_t *_resolver = nullptr) const;
  void UnPackTo(ArrayTableT *_o, const flatbuffers::resolver_function_t *_resolver = nullptr) const;
  static flatbuffers::Offset<ArrayTable> Pack(flatbuffers::FlatBufferBuilder &_fbb, const ArrayTableT* _o, const flatbuffers::rehasher_function_t *_rehasher = nullptr);
};

struct ArrayTableBuilder {
  typedef ArrayTable Table;
  flatbuffers::FlatBufferBuilder &fbb_;
  flatbuffers::uoffset_t start_;
  void add_a(const MyGame::Example::ArrayStruct *a) {
    fbb_.AddStruct(ArrayTable::VT_A, a);
  }
  explicit ArrayTableBuilder(flatbuffers::FlatBufferBuilder &_fbb)
        : fbb_(_fbb) {
    start_ = fbb_.StartTable();
  }
  flatbuffers::Offset<ArrayTable> Finish() {
    const auto end = fbb_.EndTable(start_);
    auto o = flatbuffers::Offset<ArrayTable>(end);
    return o;
  }
};

inline flatbuffers::Offset<ArrayTable> CreateArrayTable(
    flatbuffers::FlatBufferBuilder &_fbb,
    const MyGame::Example::ArrayStruct *a = nullptr) {
  ArrayTableBuilder builder_(_fbb);
  builder_.add_a(a);
  return builder_.Finish();
}

flatbuffers::Offset<ArrayTable> CreateArrayTable(flatbuffers::FlatBufferBuilder &_fbb, const ArrayTableT *_o, const flatbuffers::rehasher_function_t *_rehasher = nullptr);


inline bool operator==(const ArrayTableT &lhs, const ArrayTableT &rhs) {
  return
      ((lhs.a == rhs.a) || (lhs.a && rhs.a && *lhs.a == *rhs.a));
}

inline bool operator!=(const ArrayTableT &lhs, const ArrayTableT &rhs) {
    return !(lhs == rhs);
}


inline ArrayTableT::ArrayTableT(const ArrayTableT &o)
      : a((o.a) ? new MyGame::Example::ArrayStruct(*o.a) : nullptr) {
}

inline ArrayTableT &ArrayTableT::operator=(ArrayTableT o) FLATBUFFERS_NOEXCEPT {
  std::swap(a, o.a);
  return *this;
}

inline ArrayTableT *ArrayTable::UnPack(const flatbuffers::resolver_function_t *_resolver) const {
  auto _o = std::unique_ptr<ArrayTableT>(new ArrayTableT());
  UnPackTo(_o.get(), _resolver);
  return _o.release();
}

inline void ArrayTable::UnPackTo(ArrayTableT *_o, const flatbuffers::resolver_function_t *_resolver) const {
  (void)_o;
  (void)_resolver;
  { auto _e = a(); if (_e) _o->a = flatbuffers::unique_ptr<MyGame::Example::ArrayStruct>(new MyGame::Example::ArrayStruct(*_e)); }
}

inline flatbuffers::Offset<ArrayTable> ArrayTable::Pack(flatbuffers::FlatBufferBuilder &_fbb, const ArrayTableT* _o, const flatbuffers::rehasher_function_t *_rehasher) {
  return CreateArrayTable(_fbb, _o, _rehasher);
}

inline flatbuffers::Offset<ArrayTable> CreateArrayTable(flatbuffers::FlatBufferBuilder &_fbb, const ArrayTableT *_o, const flatbuffers::rehasher_function_t *_rehasher) {
  (void)_rehasher;
  (void)_o;
  struct _VectorArgs { flatbuffers::FlatBufferBuilder *__fbb; const ArrayTableT* __o; const flatbuffers::rehasher_function_t *__rehasher; } _va = { &_fbb, _o, _rehasher}; (void)_va;
  auto _a = _o->a ? _o->a.get() : nullptr;
  return MyGame::Example::CreateArrayTable(
      _fbb,
      _a);
}

inline const flatbuffers::TypeTable *TestEnumTypeTable() {
  static const flatbuffers::TypeCode type_codes[] = {
    { flatbuffers::ET_CHAR, 0, 0 },
    { flatbuffers::ET_CHAR, 0, 0 },
    { flatbuffers::ET_CHAR, 0, 0 }
  };
  static const flatbuffers::TypeFunction type_refs[] = {
    MyGame::Example::TestEnumTypeTable
  };
  static const char * const names[] = {
    "A",
    "B",
    "C"
  };
  static const flatbuffers::TypeTable tt = {
    flatbuffers::ST_ENUM, 3, type_codes, type_refs, nullptr, nullptr, names
  };
  return &tt;
}

inline const flatbuffers::TypeTable *NestedStructTypeTable() {
  static const flatbuffers::TypeCode type_codes[] = {
    { flatbuffers::ET_INT, 1, -1 },
    { flatbuffers::ET_CHAR, 0, 0 },
    { flatbuffers::ET_CHAR, 1, 0 },
    { flatbuffers::ET_LONG, 1, -1 }
  };
  static const flatbuffers::TypeFunction type_refs[] = {
    MyGame::Example::TestEnumTypeTable
  };
  static const int16_t array_sizes[] = { 2, 2, 2,  };
  static const int64_t values[] = { 0, 8, 9, 16, 32 };
  static const char * const names[] = {
    "a",
    "b",
    "c",
    "d"
  };
  static const flatbuffers::TypeTable tt = {
    flatbuffers::ST_STRUCT, 4, type_codes, type_refs, array_sizes, values, names
  };
  return &tt;
}

inline const flatbuffers::TypeTable *ArrayStructTypeTable() {
  static const flatbuffers::TypeCode type_codes[] = {
    { flatbuffers::ET_FLOAT, 0, -1 },
    { flatbuffers::ET_INT, 1, -1 },
    { flatbuffers::ET_CHAR, 0, -1 },
    { flatbuffers::ET_SEQUENCE, 1, 0 },
    { flatbuffers::ET_INT, 0, -1 },
    { flatbuffers::ET_LONG, 1, -1 }
  };
  static const flatbuffers::TypeFunction type_refs[] = {
    MyGame::Example::NestedStructTypeTable
  };
  static const int16_t array_sizes[] = { 15, 2, 2,  };
  static const int64_t values[] = { 0, 4, 64, 72, 136, 144, 160 };
  static const char * const names[] = {
    "a",
    "b",
    "c",
    "d",
    "e",
    "f"
  };
  static const flatbuffers::TypeTable tt = {
    flatbuffers::ST_STRUCT, 6, type_codes, type_refs, array_sizes, values, names
  };
  return &tt;
}

inline const flatbuffers::TypeTable *ArrayTableTypeTable() {
  static const flatbuffers::TypeCode type_codes[] = {
    { flatbuffers::ET_SEQUENCE, 0, 0 }
  };
  static const flatbuffers::TypeFunction type_refs[] = {
    MyGame::Example::ArrayStructTypeTable
  };
  static const char * const names[] = {
    "a"
  };
  static const flatbuffers::TypeTable tt = {
    flatbuffers::ST_TABLE, 1, type_codes, type_refs, nullptr, nullptr, names
  };
  return &tt;
}

inline const MyGame::Example::ArrayTable *GetArrayTable(const void *buf) {
  return flatbuffers::GetRoot<MyGame::Example::ArrayTable>(buf);
}

inline const MyGame::Example::ArrayTable *GetSizePrefixedArrayTable(const void *buf) {
  return flatbuffers::GetSizePrefixedRoot<MyGame::Example::ArrayTable>(buf);
}

inline ArrayTable *GetMutableArrayTable(void *buf) {
  return flatbuffers::GetMutableRoot<ArrayTable>(buf);
}

inline MyGame::Example::ArrayTable *GetMutableSizePrefixedArrayTable(void *buf) {
  return flatbuffers::GetMutableSizePrefixedRoot<MyGame::Example::ArrayTable>(buf);
}

inline const char *ArrayTableIdentifier() {
  return "ARRT";
}

inline bool ArrayTableBufferHasIdentifier(const void *buf) {
  return flatbuffers::BufferHasIdentifier(
      buf, ArrayTableIdentifier());
}

inline bool SizePrefixedArrayTableBufferHasIdentifier(const void *buf) {
  return flatbuffers::BufferHasIdentifier(
      buf, ArrayTableIdentifier(), true);
}

inline bool VerifyArrayTableBuffer(
    flatbuffers::Verifier &verifier) {
  return verifier.VerifyBuffer<MyGame::Example::ArrayTable>(ArrayTableIdentifier());
}

inline bool VerifySizePrefixedArrayTableBuffer(
    flatbuffers::Verifier &verifier) {
  return verifier.VerifySizePrefixedBuffer<MyGame::Example::ArrayTable>(ArrayTableIdentifier());
}

inline const char *ArrayTableExtension() {
  return "mon";
}

inline void FinishArrayTableBuffer(
    flatbuffers::FlatBufferBuilder &fbb,
    flatbuffers::Offset<MyGame::Example::ArrayTable> root) {
  fbb.Finish(root, ArrayTableIdentifier());
}

inline void FinishSizePrefixedArrayTableBuffer(
    flatbuffers::FlatBufferBuilder &fbb,
    flatbuffers::Offset<MyGame::Example::ArrayTable> root) {
  fbb.FinishSizePrefixed(root, ArrayTableIdentifier());
}

inline flatbuffers::unique_ptr<MyGame::Example::ArrayTableT> UnPackArrayTable(
    const void *buf,
    const flatbuffers::resolver_function_t *res = nullptr) {
  return flatbuffers::unique_ptr<MyGame::Example::ArrayTableT>(GetArrayTable(buf)->UnPack(res));
}

inline flatbuffers::unique_ptr<MyGame::Example::ArrayTableT> UnPackSizePrefixedArrayTable(
    const void *buf,
    const flatbuffers::resolver_function_t *res = nullptr) {
  return flatbuffers::unique_ptr<MyGame::Example::ArrayTableT>(GetSizePrefixedArrayTable(buf)->UnPack(res));
}

}  // namespace Example
}  // namespace MyGame

#endif  // FLATBUFFERS_GENERATED_ARRAYSTEST_MYGAME_EXAMPLE_H_
