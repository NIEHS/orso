import React from 'react';
import MetaPlot from './MetaPlot';

import './MetaPlot.css';


class SmallPersonalDataView extends React.Component {
    render(){
        return <div className="panel panel-default">
            <div className="panel-heading">
                <div className="panel-title pull-left"><a href={this.props.dataset_url}>{this.props.meta_data['name']}</a></div>
                <div className="panel-title pull-right">
                    <a href={this.props.update_url}><span className="glyphicon glyphicon-pencil"></span></a>&nbsp;
                    <a href={this.props.delete_url}><span className="glyphicon glyphicon-trash"></span></a>
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

SmallPersonalDataView.propTypes = {
    meta_data: React.PropTypes.object.isRequired,
    promoter_data: React.PropTypes.object.isRequired,
    enhancer_data: React.PropTypes.object.isRequired,
    dataset_url: React.PropTypes.string.isRequired,
    update_url: React.PropTypes.string.isRequired,
    delete_url: React.PropTypes.string.isRequired,
};

export default SmallPersonalDataView;
