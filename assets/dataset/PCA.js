import React from 'react';


class PCA extends React.Component {

    constructor(props) {
        super(props);

        var color_by = this.props.plot['color_options'][0];

        var point_visibility = []
        var point_num = this.props.plot['points'].length +
            this.props.user_data.length;
        for (var i = 0; i < point_num; i++) {
            point_visibility.push(true);
        }

        var vector_visibility = []
        var vector_num = this.props.plot['vectors'].length;
        for (var i = 0; i < vector_num; i ++) {
            vector_visibility.push(false);
        }

        this.state = {
            color_by: color_by,
            point_visibility: point_visibility,
            vector_visibility: vector_visibility,
        };
    }

    plotPCA() {

        var data = [];

        var points = this.props.plot['points'].concat(this.props.user_data);
        var x = [], y = [], z = [], names = [], colors = [];

        for (var i = 0; i < points.length; i++) {
            if (this.state.point_visibility[i]) {

                var point = points[i];

                if (point['colors'].hasOwnProperty(this.state.color_by)) {
                    var color = point['colors'][this.state.color_by];
                } else if (point['colors'].hasOwnProperty('Default')) {
                    var color = point['colors']['Default'];
                } else {
                    var color = '#A9A9A9';
                }

                x.push(point['transformed_values'][0]);
                y.push(point['transformed_values'][1]);
                z.push(point['transformed_values'][2]);
                names.push(point['dataset_name']);
                colors.push(color);
            }
        }

        data.push({
            x: x,
            y: y,
            z: z,
            mode: 'markers',
            text: names,
            point_data: point,
            hoverinfo: 'text',
            marker: {
                size: 12,
                color: colors,
                opacity: 0.8,
            },
            type: 'scatter3d',
        })

        var vectors = this.props.plot['vectors'];
        for (var i = 0; i < vectors.length; i++) {

            var vector = vectors[i];

            if (this.state.vector_visibility[i]) {
                data.push({
                    x: [0, vector['point'][0]],
                    y: [0, vector['point'][1]],
                    z: [0, vector['point'][2]],
                    type: 'scatter3d',
                    mode: 'lines+text',
                    line: {
                        width: 6,
                        color: vector['color'],
                    },
                    text: ['', vector['label']],
                });
            }
        }

        var layout = {
            showlegend: false,
            margin: {
                l: 0,
                r: 0,
                b: 0,
                t: 0,
            },
            scene: {
                xaxis:{
                    title: '',
                    showticklabels: false,
                },
                yaxis:{
                    title: '',
                    showticklabels: false,
                },
                zaxis:{
                    title: '',
                    showticklabels: false,
                },
            },
        };

        var config = {
            displaylogo: false,
            displayModeBar: false,
            modeBarButtonsToRemove: [
                'sendDataToCloud',
                'resetCameraLastSave3d',
                'hoverClosest3d',
            ],
        };

        Plotly.react('plot', data, layout, config);
    }

    drawPlotlyVariance(){

        var data = [{
            y: ['PC 1', 'PC 2', 'PC 3'].reverse(),
            x: this.props.explained_variance.reverse(),
            type: 'bar',
            orientation: 'h',
        }];

        var layout = {
            margin: {
                l: 50,
                r: 50,
                b: 50,
                t: 50,
                pad: 4,
            },
        };

        var options = {
            displayModeBar: false,
        };

        Plotly.newPlot('variance_plot', data, layout, options);
    }

    drawPlotlyComponent(component, div){
        var labels = [],
            values = [];
        for(var i = 0; i < component.length; i++){
            labels.push(component[i][0]);
            values.push(component[i][1]);
        }

        var data = [{
            y: labels.reverse(),
            x: values.reverse(),
            type: 'bar',
            orientation: 'h',
        }];

        var layout = {
            yaxis: {
                tickfont: {
                    size: 10,
                },
            },
            autosize: false,
            width: $('#tabs').width(),
            height: $('#tabs').height(),
            margin: {
                l: 50,
                r: 50,
                b: 50,
                t: 50,
                pad: 4,
            },
        };

        Plotly.newPlot(div, data, layout);
    }

    removePlotly(divElement){
        $(divElement).empty();
    }

    updatePointVisibility() {
        var tags = [];
        var inputs = $(this.refs.point_visibility_dropdown).find('input');
        for (var i = 0; i < inputs.length; i++) {
            var input = inputs[i];
            if ($(input).is(':checked')) {
                tags.push($(input).val());
            }
        }

        var points = this.props.plot['points'].concat(this.props.user_data);
        var visibilities = [];
        for (var i = 0; i < points.length; i++) {
            var point = points[i];
            var visibility = true;
            for (var j = 0; j < tags.length; j++) {
                var tag_1 = tags[j];
                for (var k = 0; k < point['tags'].length; k++) {
                    var tag_2 = point['tags'][k];
                    if (tag_1 == tag_2) {
                        visibility = false;
                    }
                }
            }
            visibilities.push(visibility);
        }

        this.setState({point_visibility: visibilities});
    }

    updateVectorVisibility() {
        var tags = [];
        var inputs = $(this.refs.vector_visibility_dropdown).find('input');
        for (var i = 0; i < inputs.length; i++) {
            var input = inputs[i];
            if ($(input).is(':checked')) {
                tags.push($(input).val());
            }
        }

        var vectors = this.props.plot['vectors'];
        var visibilities = [];
        for (var i = 0; i < vectors.length; i++) {
            var vector = vectors[i];
            var visibility = false;
            for (var j = 0; j < tags.length; j++) {
                var tag_1 = tags[j];
                for (var k = 0; k < vector['tags'].length; k++) {
                    var tag_2 = vector['tags'][k];
                    if (tag_1 == tag_2) {
                        visibility = true;
                    }
                }
            }
            visibilities.push(visibility);
        }

        this.setState({vector_visibility: visibilities});
    }

    changeColor(event){
        this.setState({color_by: event.target.value});
    }

    componentDidMount(){

        var $color_select = $(this.refs.color_select);
        for (let i in this.props.plot['color_options']) {
            $color_select.append('<option val="' + i + '">' + this.props.plot['color_options'][i] + '</option>');
        }

        var $point_visibility_dropdown = $(this.refs.point_visibility_dropdown)
        var point_tag_keys = Object.keys(this.props.plot['point_tags']);

        for (var i = 0; i < point_tag_keys.length; i++){
            var point_tag_key = point_tag_keys[i];
            var $div = $(
                '<li className="dropdown-header">' + point_tag_key + '</li>'
            );
            $point_visibility_dropdown.append($div);

            var tags = this.props.plot['point_tags'][point_tag_key];
            for (var j = 0; j < tags.length; j++) {
                var tag = tags[j];
                var $div = $(
                    '<span> \
                        <input type="checkbox" id="vector_tag_' + i + '" value="' + tag[0] + '"></input> \
                        <label style="display:inline"><span class="glyphicon glyphicon-stop" style="color:' + tag[1] + '"></span>&nbsp;' + tag[0] + '</label> \
                        </br>\
                    </span>'
                );
                $div.on('click', this.updatePointVisibility.bind(this));
                $point_visibility_dropdown.append($div);
            }

            if (i < point_tag_keys.length - 1) {
                $point_visibility_dropdown.append($('<li role="separator" class="divider"></li>'));
            }
        }

        var $vector_visibility_dropdown = $(this.refs.vector_visibility_dropdown);
        var vector_tag_keys = Object.keys(this.props.plot['vector_tags']);

        for (var i = 0; i < vector_tag_keys.length; i++){
            var vector_tag_key = vector_tag_keys[i];
            var $div = $(
                '<li className="dropdown-header">' + vector_tag_key + '</li>'
            );
            $vector_visibility_dropdown.append($div);

            var tags = this.props.plot['vector_tags'][vector_tag_key];
            for (var j = 0; j < tags.length; j++) {
                var tag = tags[j];
                var $div = $(
                    '<span> \
                        <input type="checkbox" id="vector_tag_' + i + '" value="' + tag[0] + '"></input> \
                        <label style="display:inline"><span class="glyphicon glyphicon-stop" style="color:' + tag[1] + '"></span>&nbsp;' + tag[0] + '</label> \
                        </br>\
                    </span>'
                );
                $div.on('click', this.updateVectorVisibility.bind(this));
                $vector_visibility_dropdown.append($div);
            }

            if (i < vector_tag_keys.length - 1) {
                $vector_visibility_dropdown.append($('<li role="separator" class="divider"></li>'));
            }
        }

        this.plotPCA();
        this.drawPlotlyVariance();
        this.drawPlotlyComponent(this.props.components[0], 'pc_1');
        this.drawPlotlyComponent(this.props.components[1], 'pc_2');
        this.drawPlotlyComponent(this.props.components[2], 'pc_3');
    }

    componentDidUpdate(){
        this.plotPCA();
    }

    cleanDiv(div_id){
        $('#' + div_id).empty();
    }

    componentWillUnmount(){
        this.cleanDiv('plot');
        this.cleanDiv('variance_plot');
        this.cleanDiv('pc_1');
        this.cleanDiv('pc_2');
        this.cleanDiv('pc_3');
    }

    render(){

        return <div>
            <div className='pca' style={{paddingTop: '20px'}}>
                <h2>Plot</h2>
                <div className='row'>
                    <div className='col-sm-9' style={{border: '1px solid black'}}>
                        <div ref='plot' id='plot'></div>
                    </div>
                    <div className='col-sm-3'>
                        <div className='form-group'>
                            <label htmlFor='color_select'>Color by:</label>
                            <select className='form-control'
                                id='color_select'
                                ref='color_select'
                                onChange={this.changeColor.bind(this)}
                                value={this.state.color_by}>
                            </select>
                        </div>
                        <div className="dropdown">
                            <button className="btn btn-default btn-block dropdown-toggle" type="button" data-toggle="dropdown">
                                Hide dataset&nbsp;
                                <span className="caret"></span>
                            </button>
                            <ul className="dropdown-menu pre-scrollable" ref="point_visibility_dropdown">
                            </ul>
                        </div>
                        <div className="dropdown">
                            <button className="btn btn-default btn-block dropdown-toggle" type="button" data-toggle="dropdown" >
                                Show vector&nbsp;
                                <span className="caret"></span>
                            </button>
                            <ul className="dropdown-menu pre-scrollable" ref="vector_visibility_dropdown">
                            </ul>
                        </div>
                    </div>
                </div>
            </div>
            <div style={{paddingTop: '20px'}}>
                <h2>Components</h2>
                <ul className='nav nav-tabs' style={{paddingTop: '10px'}}>
                    <li className='active'><a data-toggle='tab' href='#variance_plot_tab'>Variance ratios</a></li>
                    <li><a data-toggle='tab' href='#pc_1_tab'>Principle component 1</a></li>
                    <li><a data-toggle='tab' href='#pc_2_tab'>Principle component 2</a></li>
                    <li><a data-toggle='tab' href='#pc_3_tab'>Principle component 3</a></li>
                </ul>
                <div className='tab-content' id='tabs'>
                    <div id='variance_plot_tab' className='tab-pane fade in active'>
                        <div id='variance_plot'></div>
                    </div>
                    <div id='pc_1_tab' className='tab-pane fade'>
                        <div id='pc_1'></div>
                    </div>
                    <div id='pc_2_tab' className='tab-pane fade'>
                        <div id='pc_2'></div>
                    </div>
                    <div id='pc_3_tab' className='tab-pane fade'>
                        <div id='pc_3'></div>
                    </div>
                </div>
            </div>
        </div>;
    }
}

PCA.defaultProps = {
    user_data: [],
};

PCA.propTypes = {
    plot: React.PropTypes.object.isRequired,
    explained_variance: React.PropTypes.array.isRequired,
    components: React.PropTypes.array.isRequired,
    user_data: React.PropTypes.array.isRequired,
};

export default PCA;
