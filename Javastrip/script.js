

(function(){
    const spanEl = document.querySelector("main h2 span");
    const txtArr = ['Web Publisher', 'Front-End Developer', 'Web UI Designer', 'UX Designer', 'Back-End Developer'];

    let index=0;
    let currentTxt = txtArr[index].split("");

    function writeTxt(){
        spanEl.textContent += currentTxt.shift();
        if(currentTxt.length !== 0){
            setTimeout(writeTxt, Math.floor(Math.random()*100));
        }else{
            currentTxt = spanEl.textContent.split("");
            setTimeout(deleteTxt, 3000);
        }
    }

    function deleteTxt(){
        currentTxt.pop();
        spanEl.textContent = currentTxt.join("");
        if(currentTxt.length !== 0){
            setTimeout(deleteTxt, Math.floor(Math.random()*100));
        }else{
            index = (index +1) % txtArr.length;
            currentTxt = txtArr[index].split("");
            writeTxt();
        }
    }
    writeTxt();
})();

const headerEl = document.querySelector("header");
window.addEventListener('scroll', function(){
    requestAnimationFrame(scrollCheck);
});

function scrollCheck(){
    let browerScrollY = window.scrollY ? window.scrollY : window.pageYOffset;
    if(browerScrollY >0){
        headerEl.classList.add("active");
    }else{
        headerEl.classList.remove("active");
    }
}

const animationMove = function(selector){
    const targetEl = document.querySelector(selector);
    const browserScrollY = window.pageYOffset;
    const targetScrollY = targetEl.getBoundingClientRect().top + browserScrollY;
    window.scrollTo({ top: targetScrollY, behavior:'smooth'});
};

const scrollMoveEl = document.querySelectorAll("[data-animation-scroll='true']");
for(let i =0; i< scrollMoveEl.length; i++){
    scrollMoveEl[i].addEventListener('click', function(e){
        const target = this.dataset.target;
        animationMove(target);
    })
}

const dataSets = [
    {
        label: 'Trend 1',
        data: [10, 20, 30, 40, 50, 60],
        borderColor: 'red',
        borderWidth: 1
    },
    {
        label: 'Trend 2',
        data: [5, 15, 25, 35, 45, 55],
        borderColor: 'blue',
        borderWidth: 1
    },
    {
        label: 'Trend 3',
        data: [20, 30, 40, 50, 60, 70],
        borderColor: 'green',
        borderWidth: 1
    }
];

const ctx = document.getElementById('myChart').getContext('2d');
let chart = new Chart(ctx, {
    type: 'line',
    data: {
        labels: ['January', 'February', 'March', 'April', 'May', 'June'],
        datasets: [dataSets[0]]
    },
    options: {
        responsive: true,
        scales: {
            y: {
                beginAtZero: true
            }
        }
    }
});

function switchTab(index) {
    document.querySelectorAll('.tab').forEach((tab, idx) => {
        tab.classList.toggle('active', idx === index);
    });
    chart.data.datasets = [dataSets[index]];
    chart.update();
}