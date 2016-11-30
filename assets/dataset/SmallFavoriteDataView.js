import React from 'react';
import ReactDOM from 'react-dom';
import MetaPlot from './MetaPlot';

import './SmallDataView.css';


class SmallFavoriteDataView extends React.Component {

    componentDidMount(){
        let id = this.props.meta_data['id'],
            remove_favorite_url = this.props.remove_favorite_url;

		$(this.refs.button).on('click', function () {
            let favorite_count = $('#favorite_counts').html();

			$.ajax({url: remove_favorite_url});
            favorite_count = favorite_count - 1;
            $('#favorite_counts').html(favorite_count)
		});
    }

    render(){
        var id_select = 'panel_' + this.props.meta_data['id'];
        var id_css_select = '#' + id_select;

        return <div className="panel panel-default" id={id_select}>
            <div className="panel-heading">
                <div className="panel-title pull-left"><a href={this.props.dataset_url}>{this.props.meta_data['name']}</a></div>
                <div className="panel-title pull-right">
                    <button type="button" ref="button" className="panel-close-button" data-target={id_css_select} data-dismiss="alert"><span className="glyphicon glyphicon-remove"></span></button>
                </div>
                <div className="clearfix"></div>
            </div>
            <div className="panel-body">
            <div className='small_data_view'>
                <div className="row">
                    <div style={{height:"200px"}} className="col-sm-6">
                        <ul>
                            <li><b>Data type:</b> {this.props.meta_data['data_type']}</li>
                            <li><b>Cell type:</b> {this.props.meta_data['cell_type']}</li>
                            <li><b>Antibody:</b> {this.props.meta_data['antibody']}</li>
                            <li>{this.props.meta_data['strand']}</li>
                            {this.props.meta_data['description'] &&
                                <li><b>Description:</b> {this.props.meta_data['description']}</li>}
                        </ul>
                    </div>
                    <div style={{height:"150px"}} className="col-sm-3">
                        <h4 style={{textAlign:"center"}}>Promoters</h4>
                        <MetaPlot
                            data={this.props.promoter_data}
                        />
                    </div>
                    <div style={{height:"150px"}} className="col-sm-3">
                        <h4 style={{textAlign:"center"}}>Enhancers</h4>
                        <MetaPlot
                            data={this.props.enhancer_data}
                        />
                    </div>
                </div>
            </div>
            </div>
        </div>;

    }
}

SmallFavoriteDataView.propTypes = {
    meta_data: React.PropTypes.object.isRequired,
    promoter_data: React.PropTypes.object.isRequired,
    enhancer_data: React.PropTypes.object.isRequired,
    dataset_url: React.PropTypes.string.isRequired,
    remove_favorite_url: React.PropTypes.string.isRequired,
};

export default SmallFavoriteDataView;
