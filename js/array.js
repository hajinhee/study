'use strict';

// Array

// 1. Declaration
const arr1 = new Array();
const arr2 = [1, 2];

// Index position
const fruits = ['apple', 'banana'];
console.log(fruits);
console.log(fruits.length);
console.log(fruits[0]);  // apple
console.log(fruits[1]);  // banana
console.log(fruits[fruits.length - 1]); 

// 3. Looping over an array
// print all fruits
console.clear();
// a. for
for (let i=0; i < fruits.length; i++) {
    console.log(fruits[i]);
}

// b. for of -> list
for (let fruit of fruits) {
    console.log(fruit);
}

// c. forEach
fruits.forEach((fruit, index) => console.log(fruit, index))  

// d. for in -> dict
for (let i in fruits) {
    console.log(i, fruits[i])  // i: key, fruits[i]: value
}

// 4. Addrion, deletion, copy
// push: add an item to the end
fruits.push('strawberry', 'peach');
console.log(fruits);

// pop: remove an item from the end
fruits.pop();
fruits.pop();
console.log(fruits);

// unshift: add an item to the benigging
fruits.unshift('strawberry', 'peach');
console.log(fruits);

// shift: remove an item from the benigging
fruits.shift();
fruits.shift();
console.log(fruits);

// note!! shift, unshift are slower than pop, push
// splice: remove an item by index position
fruits.push('strawberry', 'peach', 'lemon');
console.log(fruits);  
fruits.splice(1, 1)
console.log(fruits);  // ['apple', 'strawberry', 'peach', 'lemon']
fruits.splice(1, 1, 'apple', 'water melon');
console.log(fruits);  

// combine two arrays
const fruits2 = ['lemon', 'kiwi']
const newFruits = fruits.concat(fruits2);
console.log(newFruits);


// 5. Searching
// indexOf: find the index
console.clear();
console.log(fruits);
console.log(fruits.indexOf('apple'));
console.log(fruits.indexOf('water melon'));
console.log(fruits.indexOf('coconut'));

// includes 
console.log(fruits.includes('water melon'));
console.log(fruits.includes('coconut'));

// lastIndexOf
console.clear();
fruits.push('apple');
console.log(fruits);
console.log(fruits.indexOf('apple'));
console.log(fruits.lastIndexOf('apple'));

