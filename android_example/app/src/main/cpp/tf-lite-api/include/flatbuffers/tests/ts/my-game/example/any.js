// automatically generated by the FlatBuffers compiler, do not modify
import { Monster as MyGame_Example2_Monster } from '../../my-game/example2/monster.js';
import { Monster } from '../../my-game/example/monster.js';
import { TestSimpleTableWithEnum } from '../../my-game/example/test-simple-table-with-enum.js';
export var Any;
(function (Any) {
    Any[Any["NONE"] = 0] = "NONE";
    Any[Any["Monster"] = 1] = "Monster";
    Any[Any["TestSimpleTableWithEnum"] = 2] = "TestSimpleTableWithEnum";
    Any[Any["MyGame_Example2_Monster"] = 3] = "MyGame_Example2_Monster";
})(Any || (Any = {}));
export function unionToAny(type, accessor) {
    switch (Any[type]) {
        case 'NONE': return null;
        case 'Monster': return accessor(new Monster());
        case 'TestSimpleTableWithEnum': return accessor(new TestSimpleTableWithEnum());
        case 'MyGame_Example2_Monster': return accessor(new MyGame_Example2_Monster());
        default: return null;
    }
}
export function unionListToAny(type, accessor, index) {
    switch (Any[type]) {
        case 'NONE': return null;
        case 'Monster': return accessor(index, new Monster());
        case 'TestSimpleTableWithEnum': return accessor(index, new TestSimpleTableWithEnum());
        case 'MyGame_Example2_Monster': return accessor(index, new MyGame_Example2_Monster());
        default: return null;
    }
}
