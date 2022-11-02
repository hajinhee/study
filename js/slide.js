let pages = 0;  // 현재 인덱스 번호
let positionValue = 0;  // images 위치값
let imgCount = 2;
const IMAGE_WIDTH = 350;  // 한번 이동 시 IMAGE_WIDTH만큼 이동
const backBtn = document.querySelector(".back")
const nextBtn = document.querySelector(".next")
const images = document.querySelector(".images")

window.onload = function(){
    makedots(imgCount);
}

function makedots(imgCount){
    for(let i=0; i<=imgCount; i++){
        dot_name = "dot a"+i;
        document.getElementById("dot").innerHTML += `<div class="${dot_name}" id="${dot_name}"></div>`
        if(i==0)
            document.getElementById("dot a"+i).style= "background:#40E0D0; margin-left:"+(15*i)+"px";
        else
            document.getElementById(dot_name).style= "background:#DCDCDC; margin-left:"+(15*i)+"px";
    }
}

function change_color(pages){
    for(let i=0; i<=imgCount; i++){
        if(i==pages)
            document.getElementById("dot a"+i).style= "background:#40E0D0; margin-left:"+(15*i)+"px";
        else
            document.getElementById("dot a"+i).style= "background:#DCDCDC; margin-left:"+(15*i)+"px";
    }
}

function next() {
  if (pages < imgCount) {
    backBtn.removeAttribute('disabled')  
    positionValue -= IMAGE_WIDTH;  // IMAGE_WIDTH의 증감을 positionValue에 저장한다.
    images.style.transform = `translateX(${positionValue}px)`;  
    pages += 1;  // 다음 페이지로 이동해서 pages를 1증가 시킨다.
    change_color(pages)
  }
  if (pages === imgCount) { 
    nextBtn.setAttribute("disabled", "true");

  }
    // 마지막 장일 때 next버튼이 disabled된다.
}

function back() {
  if (pages > 0) {
    nextBtn.removeAttribute('disabled')
    positionValue += IMAGE_WIDTH;
    images.style.transform = `translateX(${positionValue}px)`;  // transform 속성의 값으로 사용되며 요소를 x축으로 이동할 수 있다.
    pages -= 1;  //이전 페이지로 이동해서 pages를 1감소 시킨다.
    change_color(pages)
  }
  if (pages === 0) {
    backBtn.setAttribute("disabled", "true"); //첫번째 장일 때 back버튼이 disabled된다.
  }
}

function init() {  //초기 화면 상태
  backBtn.setAttribute('disabled', 'true');  //속성이 disabled가 된다.
  backBtn.addEventListener("click", back);  //클릭시 다음으로 이동한다.
  nextBtn.addEventListener("click", next);  //클릭시 이전으로 이동한다.
}
init();


document.getElementById("modal_up").addEventListener('click', () => {
    document.getElementById("auth_modal").style.display = "flex";
});

document.getElementById("btn-close").addEventListener('click', () => {
    document.getElementById("auth_modal").style.display = "none";
})