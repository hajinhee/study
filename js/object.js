// Objects

// 1. Literals and properties
const obj1 = {}  // 'object literal' syntax
const obj2 = new Object();  // 'object constructor' syntax
function print(person) {
    console.log(person.name);
    console.log(person.age);
}

const ellie = {name: 'ellie', age: 4};
print(ellie);

ellie.hasjob = true;
console.log(ellie.hasjob);

delete ellie.hasjob;
console.log(ellie.hasjob);

// 2. Computed properties
// key should be always string
console.log(ellie.name);  
console.log(ellie['name']);  
ellie['hasJob'] = true;
console.log(ellie.hasjob);

function printValue(obj, key) {
    console.log(obj[key]);
}
printValue(ellie, 'name');
printValue(ellie, 'age');

// 3. Property value shorthand
const person = new Person('ellie', 30);
console.log(person);

// 4. Constructor Function
function Person(name, age) {
    // this = {};
    this.name = name;
    this.age = age;
    // return this;
}

// 5. in operator: property existence check (key in obj)
console.log('name' in ellie);
console.log('age' in ellie);
console.log('random' in ellie);

// 6. for..in vs for..of
// for (key in obj)
console.clear();
for (key in ellie) {
    console.log(key)
}

// for (value of iterale)
const array = [1, 2, 4, 5];
for (let i = 0; i < array.length; i++) {
    console.log(array[i])
}

for (value of array) {
    console.log(value)
}

// 7. Fun cloning
// Object.assign(deest, [obj1, obj2, obj3...])
const user = {name: 'ellie', age: '20'}
const user2 = user;
user2.name = 'coder';
console.log(user)

// old way
const user3 = {};
for (key in user) {
    user3[key] = user[key];
}
console.clear();
console.log(user3);

// Object.assign
const user4 = Object.assign({}, user);
console.log(user4);

// another example
const fruit1 = { color: 'red'};
const fruit2 = { color: 'blue', size: 'big'};
const mixed = Object.assign({}, fruit1, fruit2);
console.log(mixed.color);
console.log(mixed.size);

