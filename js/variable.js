// 1. Use strict
// added in ES 5
// use this for Valina Javascript.
'use strict';


// 2. Variable
// let (added in ES6)

let globalName = 'global name';
{
    let name = 'jinhee';
    console.log(name);
    name = 'hello';
    console.log(name)
    console.log(globalName)
}
console.log(name)
console.log(globalName)

// var 
// var hoisting (mave declaration from bottom to top) => 어디에 선언했냐에 상관없이 항상 제일 위로 선언을 끌어올려주는 것
// has no block scope
console.log(age)
age = 4;
console.log(age)
var age;

// 3. Contant, r(read only)
// use const whenever possiblee.
// only use let if valiable needs to change.
const daysInweek = 7;
const maxNumber = 5;

// favor immutable data type always for a few reasons:
// - security
// - thread safety
// - reduce human mistakes

// 4. Variable types
// primitive, single item: numver, string, boolean, null, undefiedn, Symbol
// object, box container 
// function, fist-class function 

const count = 17;
const size = 17.1;
console.log(`value: ${count}, type: ${typeof count}`);
console.log(`value: ${size}, type: ${typeof size}`);

// number - special numeric values: infinity, -infinity,
const infinity =1 /0;
const negativeInfinity = -1 / 0;
const nAn = 'not a number' /2;
console.log(infinity);
console.log(negativeInfinity);
console.log(nAn);

// boolean
// false: 0, null, undefined, NaN, ''
// true: any other velue
const canRead = true;
const test = 3 < 1; // false
console.log(`value: ${canRead}, type: ${typeof canRead}`);
console.log(`value: ${test}, type: ${typeof test}`);

// null 
let nothing = null;
console.log(`value: ${nothing}, type: ${typeof nothing}`);

// undefined
let x;
console.log(`value: ${x}, type: ${typeof x}`);

// symbol, create unique identifiers for objects
const symbol1 = Symbol('id');
const symbol2 = Symbol('id');
console.log(symbol1 === symbol2);  // false
const gSymbol1 = Symbol.for('id');
const gSymbol2 = Symbol.for('id');
console.log(gSymbol1 === gSymbol2);  // true
// console.log(`value: ${symbol1}, type: ${typeof symbol1}`)  // error
console.log(`value: ${symbol1.description}, type: ${typeof symbol1}`) // value: id, type: symbol

// object, real-life object, data structure
const ellie = {name: 'ellie', age: 20};
ellie.age = 21  
console.log(ellie)  // {name: 'ellie', age: 21}

// 5. Dymanic typing: dynamically typed language
let text = 'hello';
console.log(text.charAt(0)); // h
console.log(`value: ${text}, type: ${typeof text}`);  // value: hello, type: string
text = 1;
console.log(`value: ${text}, type: ${typeof text}`);  // value: 1, type: number
text = '7' + 5;
console.log(`value: ${text}, type: ${typeof text}`);  // value: 75, type: string
text = '8' / '2';
console.log(`value: ${text}, type: ${typeof text}`);  // value: 4, type: number


