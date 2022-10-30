// JSON
// JavaScript Object Notation

// 1. Object to JSON
// stringify(obj)
let json = JSON.stringify(true);
console.log(json);

json = JSON.stringify(['apple', 'banana']);
console.log(json);  // ["apple","banana"] => JSON 형태

const rabbit = {
    name: 'tori',
    color: 'white',
    size: null,
    birthDate: new Date(),
    jump: () => {
        console.log(`${this.name} can jump!`);
    },
};

json = JSON.stringify(rabbit);  // Obj -> JSON
console.log(rabbit);  // {"name":"tori","color":"white","size":null,"birthDate":"2022-10-30T14:12:35.502Z"}
// 함수 jump와 자바스크립트 자체에 들어있는 symbol과 같은 데이터는 JSON에 포함되지 않는다. 

json = JSON.stringify(rabbit, ['name', 'color', 'size']);  // 원하는 데이터만 선택하여 JSON으로 만들 수 있다.
console.log(json);  // {"name":"tori","color":"white","size":null}

json = JSON.stringify(rabbit, (key, value) => {
    console.log(`key: ${key}, value: ${value}`);
    return key === 'name' ? 'ellie' : value;  // key로 name이 들어오면 ellie로 반환
}); 
console.log(json);  


// 2. JSON to Object
// parse(json)
// console.clear();
json = JSON.stringify(rabbit);
console.log(json, typeof json);  // {"name":"tori","color":"white","size":null,"birthDate":"2022-10-30T14:26:08.443Z"} string
const obj = JSON.parse(json, (key, value) => {
    console.log(`key: ${key}, value: ${value}`);
    return key === 'birthDate' ? new Date(value) : value;
});
console.log(obj, typeof obj);  // {name: 'tori', color: 'white', size: null, birthDate: '2022-10-30T14:26:33.574Z'} 'object'
rabbit.jump();
// obj.jump(); 

console.log(rabbit.birthDate.getDate());
console.log(obj.birthDate.getDate());