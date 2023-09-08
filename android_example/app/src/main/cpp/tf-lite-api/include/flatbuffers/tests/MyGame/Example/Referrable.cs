// <auto-generated>
//  automatically generated by the FlatBuffers compiler, do not modify
// </auto-generated>

namespace MyGame.Example
{

using global::System;
using global::System.Collections.Generic;
using global::Google.FlatBuffers;

public struct Referrable : IFlatbufferObject
{
  private Table __p;
  public ByteBuffer ByteBuffer { get { return __p.bb; } }
  public static void ValidateVersion() { FlatBufferConstants.FLATBUFFERS_22_10_26(); }
  public static Referrable GetRootAsReferrable(ByteBuffer _bb) { return GetRootAsReferrable(_bb, new Referrable()); }
  public static Referrable GetRootAsReferrable(ByteBuffer _bb, Referrable obj) { return (obj.__assign(_bb.GetInt(_bb.Position) + _bb.Position, _bb)); }
  public void __init(int _i, ByteBuffer _bb) { __p = new Table(_i, _bb); }
  public Referrable __assign(int _i, ByteBuffer _bb) { __init(_i, _bb); return this; }

  public ulong Id { get { int o = __p.__offset(4); return o != 0 ? __p.bb.GetUlong(o + __p.bb_pos) : (ulong)0; } }
  public bool MutateId(ulong id) { int o = __p.__offset(4); if (o != 0) { __p.bb.PutUlong(o + __p.bb_pos, id); return true; } else { return false; } }

  public static Offset<MyGame.Example.Referrable> CreateReferrable(FlatBufferBuilder builder,
      ulong id = 0) {
    builder.StartTable(1);
    Referrable.AddId(builder, id);
    return Referrable.EndReferrable(builder);
  }

  public static void StartReferrable(FlatBufferBuilder builder) { builder.StartTable(1); }
  public static void AddId(FlatBufferBuilder builder, ulong id) { builder.AddUlong(0, id, 0); }
  public static Offset<MyGame.Example.Referrable> EndReferrable(FlatBufferBuilder builder) {
    int o = builder.EndTable();
    return new Offset<MyGame.Example.Referrable>(o);
  }

  public static VectorOffset CreateSortedVectorOfReferrable(FlatBufferBuilder builder, Offset<Referrable>[] offsets) {
    Array.Sort(offsets,
      (Offset<Referrable> o1, Offset<Referrable> o2) =>
        new Referrable().__assign(builder.DataBuffer.Length - o1.Value, builder.DataBuffer).Id.CompareTo(new Referrable().__assign(builder.DataBuffer.Length - o2.Value, builder.DataBuffer).Id));
    return builder.CreateVectorOfTables(offsets);
  }

  public static Referrable? __lookup_by_key(int vectorLocation, ulong key, ByteBuffer bb) {
    Referrable obj_ = new Referrable();
    int span = bb.GetInt(vectorLocation - 4);
    int start = 0;
    while (span != 0) {
      int middle = span / 2;
      int tableOffset = Table.__indirect(vectorLocation + 4 * (start + middle), bb);
      obj_.__assign(tableOffset, bb);
      int comp = obj_.Id.CompareTo(key);
      if (comp > 0) {
        span = middle;
      } else if (comp < 0) {
        middle++;
        start += middle;
        span -= middle;
      } else {
        return obj_;
      }
    }
    return null;
  }
  public ReferrableT UnPack() {
    var _o = new ReferrableT();
    this.UnPackTo(_o);
    return _o;
  }
  public void UnPackTo(ReferrableT _o) {
    _o.Id = this.Id;
  }
  public static Offset<MyGame.Example.Referrable> Pack(FlatBufferBuilder builder, ReferrableT _o) {
    if (_o == null) return default(Offset<MyGame.Example.Referrable>);
    return CreateReferrable(
      builder,
      _o.Id);
  }
}

public class ReferrableT
{
  [Newtonsoft.Json.JsonProperty("id")]
  [Newtonsoft.Json.JsonIgnore()]
  public ulong Id { get; set; }

  public ReferrableT() {
    this.Id = 0;
  }
}


}
