import React from 'react';
import ReactDOM from 'react-dom';

import PCA from './PCA';


class Explore extends React.Component {

    constructor(props) {
        super(props);

        var assembly_choices = (['--']).concat(Object.keys(this.props.available_exp_types));
        var exp_type_choices = (['--']).concat(Object.keys(this.props.available_assemblies));

        this.state = {
            assembly: '--',
            exp_type: '--',
            assembly_choices: assembly_choices,
            exp_type_choices: exp_type_choices,
        };
    }

    componentDidMount(){
        this.update_assembly_select();
        this.update_exp_type_select();
    }

    update_assembly_select(){
        var $assembly_select = $(this.refs.assembly_select);
        var prev_selected = $assembly_select.val();

        $assembly_select.empty();
        for (let i in this.state.assembly_choices) {
            $assembly_select.append(
                '<option val="' + i + '">' + this.state.assembly_choices[i] + '</option>');
        }

        var index = $.inArray(prev_selected, this.state.assembly_choices);
        if (index == -1) {index = 0;}
        $assembly_select.val(this.state.assembly_choices[index]);
        this.setState({assembly: $assembly_select.val()});
    }

    update_exp_type_select(){
        var $exp_type_select = $(this.refs.exp_type_select);
        var prev_selected = $exp_type_select.val();

        $exp_type_select.empty();
        for (let i in this.state.exp_type_choices) {
            $exp_type_select.append(
                '<option val="' + i + '">' + this.state.exp_type_choices[i] + '</option>');
        }

        var index = $.inArray(prev_selected, this.state.exp_type_choices);
        if (index == -1) {index = 0;}
        $exp_type_select.val(this.state.exp_type_choices[index]);
        this.setState({exp_type: $exp_type_select.val()});
    }

    change_assembly(event){
        this.setState({
            assembly: event.target.value,
            exp_type_choices: this.props.available_exp_types[event.target.value],
        }, this.update_exp_type_select);
    }

    change_exp_type(event){
        this.setState({
            exp_type: event.target.value,
            assembly_choices: this.props.available_assemblies[event.target.value],
        }, this.update_assembly_select);
    }

    clear_pca() {
        $(this.refs.pca_container).empty();
    }

    get_pca(event){
        if (this.state.assembly != '--' && this.state.exp_type != '--') {
            this.clear_pca()
            var pca_pk = this.props.pca_lookup[this.state.assembly][this.state.exp_type];
            var pca_url = `/network/api/pca-plot/${pca_pk}/`;

            var cb = function(data) {
                ReactDOM.render(
                    <PCA
                        plot={data.pca_plot}
                        explained_variance={data.explained_variance}
                        components={data.components}
                    />,
                    this.refs.pca_container,
                );
            };

            $.get(pca_url, cb.bind(this));
        }
    }

    render(){
        return <div ref='explore_container'>
            <div ref='selection_container'>
                <select ref='assembly_select'
                    onChange={this.change_assembly.bind(this)}
                    value={this.state.assembly}>
                </select>
                <select ref='exp_type_select'
                    onChange={this.change_exp_type.bind(this)}
                    value={this.state.exp_type}>
                </select>
                <button onClick={this.get_pca.bind(this)}>
                    Go
                </button>
            </div>
            <div ref='pca_container'>
            </div>
        </div>
    }
}

Explore.propTypes = {
    pca_lookup: React.PropTypes.object.isRequired,
    available_exp_types: React.PropTypes.object.isRequired,
    available_assemblies: React.PropTypes.object.isRequired,
};

export default Explore;
