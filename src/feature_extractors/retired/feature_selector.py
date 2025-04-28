class FeatureSelector:
    """Helper class to filter out low-information features"""
    
    @staticmethod
    def filter_sparse_features(df, min_non_zero_ratio=0.01):
        """Filter out features that are mostly zeros"""
        if df.empty:
            return df
            
        # Calculate proportion of non-zero values for each feature
        non_zero_ratios = (df != 0).mean()
        
        # Get features that have sufficient non-zero values
        keep_features = non_zero_ratios[non_zero_ratios >= min_non_zero_ratio].index.tolist()
        
        if len(keep_features) == 0:
            # Keep at least one feature if all are sparse
            top_feature = non_zero_ratios.nlargest(1).index.tolist()
            return df[top_feature]
            
        return df[keep_features]
    
    @staticmethod
    def filter_low_variance_features(df, min_variance=0.001):
        """Filter out features with very low variance"""
        if df.empty:
            return df
            
        # Calculate variance for each feature
        variances = df.var()
        
        # Get features with sufficient variance
        keep_features = variances[variances >= min_variance].index.tolist()
        
        if len(keep_features) == 0:
            # Keep at least one feature if all have low variance
            top_feature = variances.nlargest(1).index.tolist()
            return df[top_feature]
            
        return df[keep_features]
    
    @staticmethod
    def get_most_important_features(df, n=100):
        """Keep only the most variable features, up to n"""
        if df.empty or df.shape[1] <= n:
            return df
            
        variances = df.var()
        top_features = variances.nlargest(n).index.tolist()
        return df[top_features]