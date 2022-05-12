gitalkConfig = {
    clientID: "5b67a155acb256a4fb8a",
    clientSecret: "e30fc850ad631aeb76cc280f7e2ebe437ad427b8",
    repo: "til",
    owner: "fakeYanss",
    admin: ["fakeYanss"],
    perPage: 20,
    language: "en",
    labels: ["Gitalk"],
    distractionFreeMode: false
};

window.$docsify = {
    name: 'Peter Liu的笔记本',
    subMaxLevel: 3,
    loadNavbar: true,
    mergeNavbar: true,
    loadSidebar: true,
    //logo: 'https://foreti.me/favicon.ico',
    formatUpdated: '{YYYY}-{MM}-{DD}',
    search: 'auto',
    // ga: 'UA-112380656-1',
    markdown: {
        renderer: {
            code: function (code, lang) {
                if (lang === "mermaid") {
                    return (
                        '<div class="mermaid">' + mermaid.render('mermaid-svg-' + num++, code) + "</div>"
                    );
                }
                return this.origin.code.apply(this, arguments);
            }
        }
    },
    pagination: {
        previousText: 'PREVIOUS',
        nextText: 'NEXT',
        crossChapter: true,
        crossChapterText: true,
    },
    plugins: [
        // EditOnGithubPlugin.create("https://github.com/fakeyanss/til/blob/master/docs/"),
        function (hook, vm) {
            hook.doneEach(function () {
                var label, domObj, main, divEle, gitalk;
                label = vm.route.path.split("/").pop();
                domObj = Docsify.dom;
                main = domObj.getNode("#main");

                /**
                 * render gittalk
                 */
                if (vm.route.path.includes("zh-cn")) {
                    gitalkConfig.language = "zh-CN";
                }
                Array.apply(
                    null,
                    document.querySelectorAll("div.gitalk-container")
                ).forEach(function (ele) {
                    ele.remove();
                });
                divEle = domObj.create("div");
                divEle.id = "gitalk-container-" + label;
                divEle.className = "gitalk-container";
                divEle.style = "width: " + main.clientWidth + "px; margin: 0 auto 20px;";
                domObj.appendTo(domObj.find(".content"), divEle);
                gitalk = new Gitalk(
                    Object.assign(gitalkConfig, {
                        id: !label ? "home" : label
                    })
                );
                gitalk.render("gitalk-container-" + label);
            })
        }
    ]
}

// move theme button to sidebar
function move() {
    document.getElementsByClassName('sidebar')[0].setAttribute('id', 'sidebar-full')
    document.getElementById('sidebar-full').appendChild(document.getElementById('docsify-darklight-theme'))
    var button = document.getElementById('docsify-darklight-theme')
    button.style.position = 'fixed'
    button.style.left = '10px'
    button.style.bottom = '58px'
    button.style.top = 'auto'
    button.style.width = '18px'
    button.style.height = '18px'
}
var refreshId = setInterval(function () {
    if (null != document.getElementById('docsify-darklight-theme')) {
        move();
        clearInterval(refreshId);
    }
}, 100);

if (typeof navigator.serviceWorker !== 'undefined') {
    navigator.serviceWorker.register('./dist/sw.js')
}