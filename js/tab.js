let faq_ul = document.getElementsByClassName("nav")[0];
let tabIndicator = document.getElementsByClassName("tab-indicator")[0];
let tabBody = document.getElementsByClassName("tab-body")[0];
let tabSpane = faq_ul.getElementsByTagName("li");

for (let i = 0; i < tabSpane.length; i++) {
  tabSpane[i].addEventListener("click", function () {
    faq_ul.getElementsByClassName("active")[0].classList.remove("active");
    tabSpane[i].classList.add("active");

    tabBody.getElementsByClassName("active")[0].classList.remove("active");
    tabBody.getElementsByTagName("div")[i].classList.add("active");

    tabIndicator.style.left = `calc(calc(100% / 5) * ${i})`;  //0%, 20%, 40%, 60%, 80%
  });
}

