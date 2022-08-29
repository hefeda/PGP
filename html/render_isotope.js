// Gauge examples: https://www.cssscript.com/demo/animated-svg-gauge/

let $grid = $('#isotope-container');

function renderIsotope(batchData) {
    console.log(batchData);

    for(let index=0; index<batchData.ids.length; index++){
        let sequenceLength = batchData.seqs[index].length

        let aggregate = {};

        aggregate["order"] = batchData.disorder[index].filter(e => e>8).length
        aggregate["disorder"] = batchData.disorder[index].filter(e => e<8).length
        aggregate["helix"] = batchData.DSSP3[index].filter(e => e==="H").length
        aggregate["sheet"] = batchData.DSSP3[index].filter(e => e==="L").length
        // aggregate["signalPeptides"] = batchData.tmbed[index].join("").match("S+")



        let elementDiv = $("<div>")
            .addClass("ui segment")
            .attr('data-sheet', (aggregate['sheet']/sequenceLength)*100)
            .attr('data-disorder', (aggregate['disorder']/sequenceLength)*100)
            .attr('data-helix', (aggregate['helix']/sequenceLength)*100)
        ;

        let gridDiv = $("<div>")
            .addClass("ui stackable grid")
        ;
        elementDiv.append(gridDiv);

        let rowDiv = $("<div>")
            .addClass("row")
        ;
        gridDiv.append(rowDiv);

        let titleDiv = $("<div>")
            .html(batchData.ids[index])
            .addClass("column four wide")
        ;
        rowDiv.append(titleDiv);

        let sequenceLengthGauge = document.createElement("div");
        sequenceLengthGauge.className = "gauge";
        Gauge(sequenceLengthGauge, {
            max: sequenceLength,
            value: sequenceLength,
            dialStartAngle: 0,
            dialEndAngle: 0,
        });

        let disorderGauge = document.createElement("div");
        disorderGauge.className = "gauge";
        Gauge(disorderGauge, {
            max: 100,
            value: (aggregate['disorder']/sequenceLength)*100,
        });

        let alphaGauge = document.createElement("div");
        alphaGauge.className = "gauge";
        Gauge(alphaGauge, {
            max: 100,
            value: (aggregate['helix']/sequenceLength)*100,
        });

        let sheetGauge = document.createElement("div");
        sheetGauge.className = "gauge";
        Gauge(sheetGauge, {
            max: 100,
            value: (aggregate['sheet']/sequenceLength)*100,
        });

        rowDiv.append(labelAndDiv("Sequence length", sequenceLengthGauge));
        rowDiv.append(labelAndDiv("Disorder", disorderGauge));
        rowDiv.append(labelAndDiv("Helix", alphaGauge));
        rowDiv.append(labelAndDiv("Sheet", sheetGauge));

        $grid.append(elementDiv);
    }

    $grid.isotope({
        getSortData: {
            sheet: '[data-sheet]',
            helix: '[data-helix]',
            disorder: '[data-disorder]',
        }
    });

    setTimeout(() => {
        $grid.isotope({ sortBy: 'sheet' })
    }, 1000)

}

function labelAndDiv(label, value){
    let columnDiv = $("<div>")
        .addClass("column two wide")
    ;
    let gridDiv = $("<div>")
        .addClass("ui statistic")
    ;
    let labelRow = $("<div>")
        .addClass("label")
        .html(label)
    ;
    let valueRow = $("<div>")
        .addClass("value")
    ;

    columnDiv.append(gridDiv);
    valueRow.append(value);
    gridDiv.append(valueRow);
    gridDiv.append(labelRow);

    return columnDiv;
}

// function labelAndDiv(label, value){
//     let columnDiv = $("<div>")
//         .addClass("column three wide")
//     ;
//     let gridDiv = $("<div>")
//         .addClass("ui grid")
//     ;
//     let labelRow = $("<div>")
//         .addClass("row centered")
//         .html(label)
//     ;
//     let valueRow = $("<div>")
//         .addClass("row centered")
//     ;
//
//     columnDiv.append(gridDiv);
//     gridDiv.append(labelRow);
//     valueRow.append(value);
//     gridDiv.append(valueRow);
//
//     return columnDiv;
// }


// let gauge0 = Gauge(document.getElementById("gauge0"));