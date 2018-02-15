import React from 'react';


class PCA extends React.Component {

    constructor(props) {
        super(props);

        var color_keys = Object.keys(this.props.plot['colors']).sort();
        var color_by_choices = ['None'];
        for (var i = 0; i < color_keys.length; i++) {
            if (color_keys[i] != 'None') {
                color_by_choices.push(color_keys[i]);
            }
        }

        this.state = {
            color_by: 'None',
            color_by_choices: color_by_choices,
        };
    }

    datasetUrl(dataset_pk) {
        return `/network/dataset/${dataset_pk}/`;
    }

    experimentUrl(experiment_pk) {
        return `/network/experiment/${experiment_pk}/`;
    }

    colorScale(index) {
        var color_scale = d3.scale.category20().domain(d3.range(0,20));
        return color_scale(index % 20);
    }

    drawPlotlyPCA() {
        var data = this.props.plot['points'];

        var pca = [], cell_types = [], targets = [], names = [];
        for (var i = 0; i < data.length; i++) {
            pca[i] = {
                x: data[i]['transformed_values'][0],
                y: data[i]['transformed_values'][1],
                z: data[i]['transformed_values'][2],
            };
            if (data[i]['experiment_target'] == '') {
                names[i] = `Cell type: ${data[i]['experiment_cell_type']}`;
            } else {
                names[i] = `Cell type: ${data[i]['experiment_cell_type']}
                            <br>Target: ${data[i]['experiment_target']}`;
            }
        }

        var colors = this.props.plot['colors'][this.state.color_by];

        function unpack(rows, key) {
            return rows.map(function(row)
                { return row[key]; });
        }
        var trace1 = {
            x: unpack(pca, 'x'),
            y: unpack(pca, 'y'),
            z: unpack(pca, 'z'),
            mode: 'markers',
            text: names,
            point_data: data,
            hoverinfo: 'text',
            marker: {
                size: 12,
                color: colors,
                opacity: 0.8,
            },
            type: 'scatter3d',
        };

        var data = [trace1];
        var layout = {
            height: 800,
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
        var options = {
            displaylogo: false,
            displayModeBar: false,
            modeBarButtonsToRemove: [
                'sendDataToCloud',
                'resetCameraLastSave3d',
                'hoverClosest3d',
            ],
        };
        Plotly.newPlot('plot', data, layout, options);

        var plot = document.getElementById('plot');
        plot.on('plotly_click', function(data){
            for(var i = 0; i < data.points.length; i++){
                var index = data.points[i].pointNumber;
                var selected_object = data.points[i].data.point_data[index];
                this.setState({'selected_object': selected_object}, this.setSelection);
            }
        }.bind(this));
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

    setSelection(){
        $(this.refs.selection_exp_name).text(this.state.selected_object.experiment_name);
        $(this.refs.selection_ds_name).text(this.state.selected_object.dataset_name);
        $(this.refs.selection_cell_type).text(this.state.selected_object.experiment_cell_type);

        if (this.state.selected_object.experiment_target == '') {
            $(this.refs.selection_target).text('--');
        } else {
            $(this.refs.selection_target).text(this.state.selected_object.experiment_target);
        }

        $(this.refs.experiment_link).attr('href', this.experimentUrl(this.state.selected_object.experiment_pk));
        $(this.refs.dataset_link).attr('href', this.datasetUrl(this.state.selected_object.dataset_pk));
        $(this.refs.experiment_link).attr('target', '_blank');
        $(this.refs.dataset_link).attr('target', '_blank');
    }

    updateColor() {
        var colors = this.props.plot['colors'][this.state.color_by];

        var update = {
            marker: {
                size: 12,
                color: colors,
                opacity: 0.8,
            }
        };
        Plotly.update('plot', update);
    }

    componentDidMount(){

        var $color_by_select = $(this.refs.color_by_select);
        for (let i in this.state.color_by_choices) {
            $color_by_select.append(
                '<option val="' + i + '">' + this.state.color_by_choices[i] + '</option>');
        }
        this.drawPlotlyPCA();
        this.drawPlotlyVariance();
        this.drawPlotlyComponent(this.props.components[0], 'pc_1');
        this.drawPlotlyComponent(this.props.components[1], 'pc_2');
        this.drawPlotlyComponent(this.props.components[2], 'pc_3');
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

    change_color(event){
         this.setState({color_by: event.target.value}, this.updateColor);
    }

    render(){
        return <div className='pca'>
            <div className='row'>
                <div className='col-sm-8'>
                    <div ref='plot' id='plot'></div>
                </div>
                <div className='col-sm-4'>
                    <div>Color by</div>
                    <select ref='color_by_select'
                        onChange={this.change_color.bind(this)}
                        value={this.state.color_by}>
                    </select>
                    <div ref='selection_container'>
                        <div><h4>Selection</h4></div>
                        <div><b>Experiment name: </b><span ref='selection_exp_name'>--</span></div>
                        <div><b>Dataset name: </b><span ref='selection_ds_name'>--</span></div>
                        <div><b>Cell/tissue type: </b><span ref='selection_cell_type'>--</span></div>
                        <div><b>Target: </b><span ref='selection_target'>--</span></div>
                        <div><a ref='experiment_link' href='#' className='btn btn-default'>Go to Experiment</a></div>
                        <div><a ref='dataset_link' href='#' className='btn btn-default'>Go to Dataset</a></div>
                    </div>
                    <ul className='nav nav-tabs'>
                        <li className='active'><a data-toggle='tab' href='#variance_plot_tab'>PCA</a></li>
                        <li><a data-toggle='tab' href='#pc_1_tab'>PC 1</a></li>
                        <li><a data-toggle='tab' href='#pc_2_tab'>PC 2</a></li>
                        <li><a data-toggle='tab' href='#pc_3_tab'>PC 3</a></li>
                    </ul>
                    <div className='tab-content' id='tabs'>
                        <div id='variance_plot_tab' className='tab-pane fade in active'>
                            <h4>Variance ratios</h4>
                            <div id='variance_plot'></div>
                        </div>
                        <div id='pc_1_tab' className='tab-pane fade'>
                            <h4>Principle component 1</h4>
                            <div id='pc_1'></div>
                        </div>
                        <div id='pc_2_tab' className='tab-pane fade'>
                            <h4>Principle component 2</h4>
                            <div id='pc_2'></div>
                        </div>
                        <div id='pc_3_tab' className='tab-pane fade'>
                            <h4>Principle component 3</h4>
                            <div id='pc_3'></div>
                        </div>
                    </div>
                </div>
            </div>
        </div>;
    }
}

PCA.propTypes = {
    plot: React.PropTypes.object.isRequired,
    explained_variance: React.PropTypes.array.isRequired,
    components: React.PropTypes.array.isRequired,
};

export default PCA;
