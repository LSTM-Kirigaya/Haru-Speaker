() => {
    let root = document.querySelector("body > gradio-app");
    if (root.shadowRoot != null) {
        root = root.shadowRoot;
    }
    let audio = root.querySelector("#tts-audio").querySelector("audio");
    let text = root.querySelector("#input-text").querySelector("textarea");
    if (audio == undefined) {
        return;
    }
    text = text.value;
    if (text == undefined) {
        text = Math.floor(Math.random() * 100000000);
    }
    audio = audio.src;
    let download_element = document.createElement("a");
    download_element.download = text.substr(0, 20) + '.wav';
    download_element.href = audio;
    document.body.appendChild(download_element);
    download_element.click();
    download_element.remove();
}