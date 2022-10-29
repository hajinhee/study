 // 1. Function declaration
 function printHello() {
    console.log('Hello');
 }
 printHello(); 

 function log(message) {
    console.log(message);
 }
 log('Hello');
 log(1234);

 // 2. Parameters
 // premitive parameters: passed by value
 // object parameters: passed by reference
 function changeName(obj) {
    obj.name = 'coder';
 } 

 const ellie = { name: 'ellie'};
 changeName(ellie);
 console.log(ellie);

 // 3. Default parameters (added in ES6)
 function showMessage(message, from = 'unknown') {
    console.log(`${message} by ${from}`);
 }
 showMessage('Hi!');

 // 4. Rest parameters (added in ES6) => 배열 형태로 전달
 function printAll(...args) {
    for (let i = 0; i < args.length; i++) {
        console.log(args[i]);
    }
    
    for (const arg of args) {
        console.log(arg);
    }

    args.forEach((arg) => console.log(arg));
 }
 printAll('dream', 'coding', 'ellie');

 // 5. Local scope
 let globalMessage = 'global';  // global variable
 function printMessage() {
    let message = 'hello';
    console.log(message);  // local variable
    console.log(globalMessage);
 }
 printMessage();

 // 6. Return a value
 function sum(a, b) {
    return a + b;
 } 
 const result = sum(1, 2);
console.log(`sum: ${sum(1, 2)}`);

// 7. Early return, early exit
// bad
function upgradeUser(user) {
    if (user.point > 10) {
        // long upgrade logic... 
    }
}

// good **********
function upgradeUser(user) {
    if (user.point <= 10) {
        return; 
    }
    // long upgrade logic... 
}

// 8. Function expression
const print = function() {
    console.log('print');
};
print();
const printAgain = print;
printAgain();
const sumAgain = sum;
console.log(sumAgain(1, 3));

// 9. Callback function using function expression
function randomQuiz(answer, printYes, printNo) {
    if (answer === 'love you') {
        printYes();
    } else {
        printNo();
    }
}
// anonymous function
const printYes = function() {
    console.log('yes!');
}
// named functionO
const printNo = function print() {
    console.log('no!');
};
randomQuiz('wrong', printYes, printNo)
randomQuiz('love you', printYes, printNo)

// Arrow function
// always anonymous function
const add = (a, b) => a + b;

const add3 = (a, b) => {
    return a + b;
};

const add2 = function(a, b) {
    return a + b;
};

// IIFE: Immediately Invoked Function Expression
(function hello() {
    console.log('IIFE');
})();