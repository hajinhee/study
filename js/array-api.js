 // Q1. make a string out of an array
 {
    const fruits = ['apple', 'banana', 'orange'];
    console.log(fruits.join());
  }

  // Q2. make an array out of a string
  {
    const fruits = 'π, π₯, π, π';
    console.log(fruits.split(','));
    console.log(typeof fruits);
  }
  
  // Q3. make this array look like this: [5, 4, 3, 2, 1]
  {
    const array = [1, 2, 3, 4, 5];
    console.log(array.reverse())
  }
  
  // Q4. make new array without the first two elements
  {
    const array = [1, 2, 3, 4, 5];
    console.log(array.slice(2, array.length))
  }
  
  class Student {
    constructor(name, age, enrolled, score) {
      this.name = name;
      this.age = age;
      this.enrolled = enrolled;
      this.score = score;
    }
  }
  const students = [
    new Student('A', 29, true, 45),
    new Student('B', 28, false, 80),
    new Student('C', 30, true, 90),
    new Student('D', 40, false, 66),
    new Student('E', 18, true, 88),
  ];
  
  // Q5. find a student with the score 90
  {
    const result = students.find((student) => student.score === 90);
    console.log(result);  // trueμΈ μ²« λ²μ§Έ κ°λ§ 
  }
  
  // Q6. make an array of enrolled students
  {
    const result = students.filter((student) => student.enrolled === true);
    console.log(result);  // μνλ κ°λ§
  }
  
  // Q7. make an array containing only the students' scores
  // result should be: [45, 80, 90, 66, 88]
  {
    const result = students.map((student) => student.score);
    console.log(result);  // λ°°μ΄ μμ μλ κ°μ μνλ λ°©μμΌλ‘
  }
  
  // Q8. check if there is a student with the score lower than 50
  {
    console.clear();
    const result = students.some((student) => student.score < 50);
    console.log(result);  // νλλΌλ ν΄λΉλλ©΄
    // return true -> break, false -> continue   

    const result2 = !students.every((student) => student.score >= 50);
    console.log(result2);  // λͺ¨λ  κ°μ΄
    // someμ λ°λ 
}
  
  // Q9. compute students' average score
  {   // reduce: λ°°μ΄ νλνλλ₯Ό λλ©΄μ κ°μ λμ 
    const result = students.reduce((prev, curr) => prev + curr.score, 0);
    console.log(result / students.length);
  }
  
  // Q10. make a string containing all the scores
  // result should be: '45, 80, 90, 66, 88'
  {
    const result = students.map(student => student.score)
    .filter((score) => score >= 50)
    .join();
    console.log(result);
  }
  
  // Bonus! do Q10 sorted in ascending order
  // result should be: '45, 66, 80, 88, 90'
  {
    const result = students
    .map((student) => student.score)
    .sort()
    .join();
    console.log(result);
  }