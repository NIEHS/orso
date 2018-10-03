from neomodel import (
    DateTimeProperty,
    FloatProperty,
    IntegerProperty,
    Relationship,
    RelationshipFrom,
    RelationshipTo,
    StructuredNode,
    StructuredRel,
    StringProperty,
)


# NODES
class GraphUser(StructuredNode):
    rel_id = IntegerProperty(unique_index=True, required=True)

    # PROPERTIES
    username = StringProperty(required=True)

    # RELATIONSHIPS
    owned_experiment = RelationshipTo('GraphExperiment', 'OWN')
    favorited_experiment = RelationshipTo('GraphExperiment', 'FAVORITE')
    recommended_experiment = \
        RelationshipFrom('GraphExperiment', 'RECOMMENDATION')

    followed_user = RelationshipTo('GraphUser', 'FOLLOW')
    following_user = RelationshipFrom('GraphUser', 'FOLLOW')


class GraphExperiment(StructuredNode):
    rel_id = IntegerProperty(unique_index=True, required=True)

    # PROPERTIES
    name = StringProperty(required=True)

    experiment_type_id = IntegerProperty(required=True)
    organism_id = IntegerProperty(required=True)

    # RELATIONSHIPS
    dataset = RelationshipFrom('GraphDataset', 'OF')

    sim_experiment = Relationship('GraphExperiment', 'SIMILARITY')

    recommended_user = RelationshipTo('GraphUser', 'RECOMMENDATION')
    owning_user = RelationshipFrom('GraphUser', 'OWN')
    favoriting_user = RelationshipFrom('GraphUser', 'FAVORITE')


class GraphDataset(StructuredNode):
    rel_id = IntegerProperty(unique_index=True, required=True)

    # PROPERTIES
    name = StringProperty(required=True)

    assembly_id = IntegerProperty(required=True)

    # RELATIONSHIPS
    sim_dataset = Relationship('GraphDataset', 'SIMILARITY')

    experiment = RelationshipTo('GraphExperiment', 'OF')


# RELATIONSHIPS
class RelFavorite(StructuredRel):
    created = DateTimeProperty(default_now=True)
    updated = DateTimeProperty(default_now=True)


class RelFollow(StructuredRel):
    created = DateTimeProperty(default_now=True)
    updated = DateTimeProperty(default_now=True)


class RelSimilarity(StructuredRel):
    created = DateTimeProperty(default_now=True)
    updated = DateTimeProperty(default_now=True)

    sim_type = StringProperty(choices={
        'M': 'metadata',
        'P': 'primary',
    }, required=True)
    score = FloatProperty(required=True)


class RelRecommendation(StructuredRel):
    created = DateTimeProperty(default_now=True)
    updated = DateTimeProperty(default_now=True)
