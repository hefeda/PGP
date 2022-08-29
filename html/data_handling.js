const { fs, configure } = window.zip;
const zip = window.zip;

let loader = document.createElement("div");
loader.classList.add("loader");

let content = document.getElementById("input-dropper");
let fileReader = new FileReader();

fileReader.addEventListener("load", function () {
    console.log(fileReader.result);
}, false);

function showLoadingGUI(){
    content.innerHTML = '';
    content.appendChild(loader);
}

function fileChange(event){
    showLoadingGUI();

    var files = event.target.files; // FileList object
    // fileReader.readAsDataURL(files[0]);
    zip.ZipReader(new zip.HttpReader(files[0]), onFileExtracted, (e) => console.error(e));
    return false;
}

function handleDragOver(event) {
    event.stopPropagation();
    event.preventDefault();
    event.dataTransfer.dropEffect = 'copy'; // Explicitly show this is a copy.
}

const residueToOutput = {
    DSSP3: "dssp3_pred.txt",
    DSSP8: "dssp8_pred.txt",
    conservation: "conservation_pred.txt",
    tmbed: "membrane_tmbed.txt",
    seqs: "seqs.txt",
}

async function reconstructFiles(zipFiles) {
    let result = {}
    for(const zipFile of zipFiles){
        for(const key of Object.keys(residueToOutput)){
            if(zipFile.filename.includes(residueToOutput[key])){
                const text = await zipFile.getData(new zip.TextWriter());
                result[key] = text.split("\n").map(e => e.split(''));
            }
        }
        if(zipFile.filename.includes("ids.txt")){
            const text = await zipFile.getData(new zip.TextWriter());
            result['ids'] = text.split("\n");
        }
        if(zipFile.filename.includes("seth_disorder_pred.csv")){
            const text = await zipFile.getData(new zip.TextWriter());
            result['disorder'] = text.split("\n").map(e => e.split(', ').map(parseFloat));
        }
    }

    renderIsotope(result);
}

async function loadExample(event){
    showLoadingGUI();

    event.stopPropagation();
    event.preventDefault();

    let zipFiles = await new zip.ZipReader(new zip.HttpReader("example_output.zip")).getEntries();
    await reconstructFiles(zipFiles);
}

function clear(){
    window.location.href = location.pathname;
}

let dropZone = document.getElementById('file');

dropZone.addEventListener('change', fileChange, false);
dropZone.addEventListener('dragover', handleDragOver, false);
dropZone.addEventListener('drop', fileChange, false);


// TODO
document.getElementById('load-example').addEventListener('click', loadExample, false);

// TODO: Use ?data=XX to load data
// let urlParams = new URLSearchParams(window.location.search);
// let dataParam = urlParams.get('data');
// if(dataParam !== undefined && dataParam !== null) {
//     showLoadingGUI();
//     EVzoom.initialize(dataParam);
// }